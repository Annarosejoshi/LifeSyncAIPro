# Local frontend dependencies

The runtime interface does not require a CDN or Google Fonts. These pinned
assets are served through Django static files:

| Package | Version | License |
| --- | --- | --- |
| Bootstrap CSS and bundled JavaScript | 5.3.8 | MIT, bootstrap/LICENSE |
| Bootstrap Icons CSS and fonts | 1.13.1 | MIT, bootstrap-icons/LICENSE |
| Chart.js UMD build | 4.5.1 | MIT, chartjs/LICENSE.md |

Original source URLs, sizes, and SHA-256 hashes are recorded in `manifest.json`.
Bootstrap files additionally match the published official SHA-384 integrity
values. The application uses installed system fonts.

To intentionally refresh these vendored files, review the pinned versions and
run `python tools/vendor_assets.py` with internet access. Installation still
requires Python dependencies; runtime frontend assets work locally afterward.
