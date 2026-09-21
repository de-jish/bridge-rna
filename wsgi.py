"""The canonical Gunicorn entry point, including captured shipment identity."""
import json
import os
from pathlib import Path

manifest = Path(__file__).resolve().with_name('ship.json')
if manifest.is_file():
    tmp = manifest.parent / '.runtime/tmp'
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TMPDIR', str(tmp))
from app import build_app

RELEASE_ID = json.loads(manifest.read_text())['id'] if manifest.is_file() else 'development'
dash_app = build_app()
application = dash_app.server


@application.after_request
def shipment_header(response):
    response.headers['X-Bridge-Release'] = RELEASE_ID
    return response


@application.get('/__release')
def shipment_identity():
    response = application.json.response({'release_id': RELEASE_ID})
    response.headers['Cache-Control'] = 'no-store'
    return response
