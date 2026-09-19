"""Transfer boundary and canonical apply regressions. Never connects to NASA."""
import hashlib
import json
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'deploy'))
import common
import ship
import apply as apply_tool
from sftp_client import SFTP, Packet, string, u32, ROOT


def test_package_is_allowlisted_and_integrity_checked(tmp_path):
    dest=ship.build(tmp_path)
    meta=common.verify_bundle(dest)
    assert 'prompts/ai_summary_prompt.txt' in meta['files']
    assert 'precompute/embed_upload.py' in meta['files']
    assert 'generate_archs4_embeddings.py' in meta['files']
    assert {str(p.relative_to(dest)) for p in dest.rglob('*.md')} == {
        'assets/fonts/HDS-LICENSE.md', 'assets/fonts/public-sans/LICENSE.md'}
    assert not list(dest.rglob('.env*'))
    assert not (dest/'deploy/ship.py').exists()
    assert not (dest/'data/osdr').exists()
    assert (dest/'requirements.txt').read_text().endswith('-r deploy/requirements.lock\n')
    (dest/'app.py').write_text('tampered')
    with pytest.raises(ValueError,match='mismatch'):
        common.verify_bundle(dest)


@pytest.mark.parametrize('path',['../bridge-rna-deploy', '/home/ubuntu/bridge-rna-deploy',
                                 '.ship/incoming/abc/../../../../escape'])
def test_sftp_rejects_boundary_escape_before_requests(path):
    s=SFTP.__new__(SFTP)
    with pytest.raises(ValueError):
        s.path(path)


@pytest.mark.parametrize('path',['.env','fm_viz_env/pyvenv.cfg','uploads/query.csv','private.json'])
def test_sftp_never_downloads_private_files(path):
    s=SFTP.__new__(SFTP)
    with pytest.raises(ValueError,match='authorized application read'):
        s.get(path)


def test_sftp_never_writes_live_code():
    s=SFTP.__new__(SFTP)
    with pytest.raises(ValueError,match='new .ship/incoming'):
        s.put('app.py',b'bad')


def test_sftp_lstat_stops_at_symlink_without_following_it():
    s=SFTP.__new__(SFTP);calls=[]
    def call(kind,data):
        path=Packet(data).string().decode();calls.append((kind,path))
        mode=stat.S_IFLNK if path.endswith('/data') else stat.S_IFDIR
        return Packet(u32(4)+u32(mode|0o755))
    s.call=call
    with pytest.raises(ValueError,match='symlink refused'):
        s.checked('data/osdr/metadata/file.tsv')
    assert calls==[(7,ROOT),(7,ROOT+'/data')]


def test_common_rejects_symlink_and_untracked_deletion(tmp_path):
    (tmp_path/'data').symlink_to('/outside')
    with pytest.raises(ValueError,match='symlink'):
        common.safe(tmp_path,'data/ensembl/orthologs_one2one.txt')
    assert not common.runtime_path('uploads/test.py')
    assert not common.runtime_path('.env')
    assert not common.runtime_path('fm_viz_env/bin/python')


def fixture_stage(tmp_path):
    root=tmp_path/'fm_viz_new';root.mkdir()
    payload=ship.build(tmp_path/'local')
    meta=json.loads((payload/'ship.json').read_text())
    stage=root/'.ship/incoming'/meta['id']
    import shutil
    stage.parent.mkdir(parents=True);shutil.copytree(payload,stage)
    (root/'app.py').write_text('previous code')
    (root/'fm_viz_env').mkdir()
    (root/'fm_viz_env/marker').write_text('original environment')
    (root/'untracked.txt').write_text('preserve')
    meta['server_before']={n:common.sha256(root/n) if (root/n).is_file() else None for n in meta['files']}
    common.atomic_json(stage/'ship.json',meta)
    return root,stage,meta


def setup_apply(monkeypatch,root):
    monkeypatch.setattr(apply_tool.platform,'system',lambda:'Linux')
    monkeypatch.setattr(apply_tool.platform,'machine',lambda:'x86_64')
    monkeypatch.setattr(apply_tool,'prepare_environment',lambda *a:(root/'fm_viz_env',False))
    monkeypatch.setattr(apply_tool,'assert_port_available',lambda:None)


def test_failed_preflight_never_stops_or_changes_production(tmp_path,monkeypatch):
    root,stage,_=fixture_stage(tmp_path);setup_apply(monkeypatch,root)
    monkeypatch.setattr(apply_tool,'preflight',lambda *a:(_ for _ in ()).throw(ValueError('bad data')))
    monkeypatch.setattr(apply_tool.processes,'stop',lambda *a:pytest.fail('must not stop'))
    with pytest.raises(ValueError,match='bad data'):
        apply_tool.apply(root,stage)
    assert (root/'app.py').read_text()=='previous code'
    assert (root/'fm_viz_env/marker').read_text()=='original environment'
    assert not (root/'.ship/transaction.json').exists()


def test_apply_failure_has_prior_state_and_restores_code(tmp_path,monkeypatch):
    root,stage,meta=fixture_stage(tmp_path);setup_apply(monkeypatch,root)
    monkeypatch.setattr(apply_tool,'preflight',lambda *a:None)
    running=[True]
    monkeypatch.setattr(apply_tool,'find_master',lambda *a:{'pid':123} if running[0] else None)
    def stop(*args):
        tx=json.loads((root/'.ship/transaction.json').read_text())
        assert tx['master']=={'pid':123}
        assert (root/tx['work']/'old-code/app.py').read_text()=='previous code'
        running[0]=False
    monkeypatch.setattr(apply_tool.processes,'stop',stop)
    original=apply_tool.copy_file
    def fail(source,dest):
        if source==stage/'wsgi.py':
            raise OSError('injected disk failure')
        original(source,dest)
    monkeypatch.setattr(apply_tool,'copy_file',fail)
    with pytest.raises(OSError,match='disk failure'):
        apply_tool.apply(root,stage)
    assert (root/'app.py').read_text()=='previous code'
    assert (root/'untracked.txt').read_text()=='preserve'
    assert (root/'fm_viz_env/marker').read_text()=='original environment'
    assert not (root/'wsgi.py').exists()
    assert not (root/'.ship/transaction.json').exists()


def test_drift_fails_before_preparation(tmp_path,monkeypatch):
    root,stage,_=fixture_stage(tmp_path);setup_apply(monkeypatch,root)
    (root/'app.py').write_text('server edit since transfer')
    monkeypatch.setattr(apply_tool,'prepare_environment',lambda *a:pytest.fail('must fail before preparation'))
    with pytest.raises(ValueError,match='changed since transfer'):
        apply_tool.apply(root,stage)


def test_success_preserves_untracked_files_and_removes_only_tracked_code(tmp_path,monkeypatch):
    root,stage,meta=fixture_stage(tmp_path);setup_apply(monkeypatch,root)
    (root/'bridge_rna').mkdir();(root/'bridge_rna/obsolete.py').write_text('old')
    common.atomic_json(root/'ship.json',{'id':'old','files':{'bridge_rna/obsolete.py':common.sha256(root/'bridge_rna/obsolete.py')}})
    meta['previous_manifest_sha256']=common.sha256(root/'ship.json')
    meta['removed']=['bridge_rna/obsolete.py']
    meta['server_before']['bridge_rna/obsolete.py']=common.sha256(root/'bridge_rna/obsolete.py')
    common.atomic_json(stage/'ship.json',meta)
    monkeypatch.setattr(apply_tool,'preflight',lambda *a:None)
    monkeypatch.setattr(apply_tool,'find_master',lambda *a:None)
    monkeypatch.setattr(apply_tool.subprocess,'Popen',lambda *a,**k:None) # observer, no service launch
    apply_tool.apply(root,stage)
    assert not stage.exists()
    assert not (root/'bridge_rna/obsolete.py').exists()
    assert (root/'untracked.txt').read_text()=='preserve'
    assert (root/'fm_viz_env/marker').read_text()=='original environment'
    assert json.loads((root/'.ship/applied.json').read_text())['status']=='applied, not confirmed running'
    assert not (root/'.ship/running.json').exists()


def test_pid_reuse_is_never_signalled(monkeypatch):
    p=apply_tool.processes
    monkeypatch.setattr(p,'process',lambda pid:{'pid':pid,'start':'new'})
    monkeypatch.setattr(p.os,'kill',lambda *a:pytest.fail('no signal'))
    with pytest.raises(ValueError,match='identity changed'):
        p.stop({}, {'process':{'pid':123,'start':'old'}})


def test_environment_install_failure_restores_original_environment(tmp_path,monkeypatch):
    root,stage,_=fixture_stage(tmp_path);setup_apply(monkeypatch,root)
    monkeypatch.setattr(apply_tool,'preflight',lambda *a:None)
    monkeypatch.setattr(apply_tool,'find_master',lambda *a:None)
    monkeypatch.setattr(apply_tool,'prepare_environment',lambda *a:(tmp_path/'candidate',True))
    monkeypatch.setattr(apply_tool,'call',lambda *a,**k:(root/'fm_viz_env').mkdir())
    monkeypatch.setattr(apply_tool,'install',lambda *a:(_ for _ in ()).throw(RuntimeError('offline installation failed')))
    with pytest.raises(RuntimeError,match='offline installation failed'):
        apply_tool.apply(root,stage)
    assert (root/'fm_viz_env/marker').read_text()=='original environment'
    assert (root/'app.py').read_text()=='previous code'
    assert not (root/'.ship/transaction.json').exists()


def test_untracked_code_cannot_be_deleted(tmp_path):
    root,stage,meta=fixture_stage(tmp_path)
    (root/'bridge_rna').mkdir();(root/'bridge_rna/untracked.py').write_text('preserve')
    meta['removed']=['bridge_rna/untracked.py']
    meta['server_before']['bridge_rna/untracked.py']=common.sha256(root/'bridge_rna/untracked.py')
    with pytest.raises(ValueError,match='not a tracked'):
        apply_tool.expected_files(root,meta)


def test_bad_transfer_readback_does_not_advance_baseline(tmp_path):
    dest=ship.build(tmp_path);meta=json.loads((dest/'ship.json').read_text())
    common.atomic_json(dest.parent/'validated.json',{'files':meta['files']})
    class FakeSFTP:
        def checked(self,*a,**k):return None
        def mkdir(self,*a):pass
        def put(self,*a):pass
        def get(self,*a):return b'transfer corruption'
    with pytest.raises(ValueError,match='transferred content mismatch'):
        ship.transfer(FakeSFTP(),dest,tmp_path,meta)
    assert not (tmp_path/'last-transferred.json').exists()
    assert not (dest.parent/'transferred.json').exists()


def test_nasa_runtime_assets_are_packaged(tmp_path):
    import re
    dest = ship.build(tmp_path)
    meta = common.verify_bundle(dest)
    for name in ("assets/00-fonts.css", "assets/00-hds-tokens.css", "assets/02-controls.css", "assets/nasa.svg"):
        assert name in meta["files"]
    for relative in re.findall(r'url\(["\']?([^"\')]+)', (dest/"assets/00-fonts.css").read_text()):
        assert "assets/" + relative in meta["files"]
    assert not any(name.startswith(("prototypes/", ".lavish/")) for name in meta["files"])


@pytest.mark.parametrize("name", ["assets/fonts/inter/private.json", "assets/fonts/inter/extra.woff2",
                                  "assets/arbitrary.svg", "assets/fonts/../private.json"])
def test_asset_packaging_does_not_broaden_private_reads(name):
    assert not common.runtime_path(name)


def test_audit_records_binary_font_differences(tmp_path):
    dest = ship.build(tmp_path)
    font = "assets/fonts/inter/Inter-Regular.woff2"
    previous = b"wOF2\x80previous-font"
    class FakeSFTP:
        def checked(self, name, missing=False):
            return {} if name == font else None
        def get(self, name):
            assert name == font
            return previous
    _, report = ship.audit(FakeSFTP(), dest, tmp_path)
    assert font in report["versus_server"]["changed"]
    assert (dest.parent/"server-before"/font).read_bytes() == previous
    diff = (dest.parent/"diffs"/font).read_text()
    assert hashlib.sha256(previous).hexdigest() in diff
    assert common.sha256(dest/font) in diff
