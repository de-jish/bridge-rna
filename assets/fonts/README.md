# NASA HDS fonts

Vendored unchanged from `@nasa-hds/core@0.10.0/dist/fonts` (exact dependency and integrity recorded in `prototypes/hds/package-lock.json`). Only the weights used by this app are included. Inter is used for headings, Public Sans Web for body text, and DM Mono for identifiers. These match HDS font-family tokens. No external font request is required.

Inter and DM Mono include their upstream licenses. Public Sans is distributed under SIL Open Font License 1.1: https://github.com/uswds/public-sans/blob/develop/LICENSE.md . HDS package license is included as HDS-LICENSE.md.

`../00-hds-tokens.css` is an unmodified copy of the package's `src/scss/base/_custom-properties.scss`. The shared app adapter is `../00-tokens.css`; no HDS or USWDS global reset is loaded.
