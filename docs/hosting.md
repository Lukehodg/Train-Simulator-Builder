# Hosting the simulator for colleagues

The viewer is a static website: the live feeds are fetched by the pipeline when the routes are built, not while
someone is using it. So hosting it "with live feeds" means two pieces:

- **GitHub Actions** (`.github/workflows/publish.yml`) rebuilds all 18 routes from the live feeds on the 17th of each
  month, or whenever you run it by hand, builds each route's Word + Excel evidence pack for the viewer's **Report**
  button, then publishes the result. Your API keys live only there, as GitHub secrets.
  GitHub's machines have open internet, so OpenStreetMap works and every route gets its exact track line.
- **Cloudflare** serves the site (a Workers static-assets site, configured in `wrangler.jsonc`; Cloudflare Pages now
  lives inside Workers), and **Cloudflare Access** decides who can open it. Colleagues open the link,
  enter their work email and type the one-time code Cloudflare emails them. No company sign-in integration is needed.

Everything below is free: Cloudflare Access for up to 50 users, and a monthly rebuild (about 5 hours) fits
within GitHub's free Actions minutes for a private repository. Setup takes about 20 minutes. Cloudflare changes its
dashboard wording from time to time, so a label may differ slightly from what is written here.

> Do the steps in this order. The site must be locked (step 3) before the first real publish (step 5).

## 1. Create a Cloudflare account and note two values

1. Sign up at <https://dash.cloudflare.com/sign-up> (free plan). Open **Zero Trust** once, pick a team name and
   the **Free** plan (it may ask for a card; you are not charged for up to 50 users).
2. **Account ID:** in the dashboard, open **Workers & Pages**; the Account ID is in the right-hand column. Keep it
   for step 4.
3. **API token:** **Manage Account** → **Account API Tokens** → **Create Token** → **Custom token**.
   - Token name: `Train Link Simulator publish`
   - Permissions: **Account** → **Workers Scripts** → **Edit** (that is all the monthly publish needs)
   - Copy the token when it is shown; it is shown only once. Keep it for step 4 and do not paste it anywhere else.

## 2. Create the site with a placeholder to get its address

The address is `https://train-link-simulator.<your-workers-subdomain>.workers.dev`. The name comes from
`wrangler.jsonc`; your workers subdomain is shown under **Workers & Pages** (you choose it the first time).

1. **Workers & Pages** → **Create** → **Upload static files** (or **Start with Hello World** and replace it), and
   name it exactly `train-link-simulator`. A single `index.html` saying `Coming soon` is enough.
   Do **not** choose **Import a repository** / connect GitHub here: Cloudflare would then publish its own copy on
   every push, straight from the repository, which has no route data (that is built by the workflow) and sits
   outside the Access rule.
2. Note the address Cloudflare shows for it.

## 3. Lock the site to your colleagues (Cloudflare Access)

1. In the dashboard sidebar open **Zero Trust**. The first time, pick a team name (anything, e.g. `yourname-tools`)
   and the **Free** plan. It may ask for a card even on the free plan; you are not charged for up to 50 users.
2. **Access** → **Applications** → **Add an application** → **Self-hosted**.
3. Application name: `Train Link Simulator`. Session duration: `1 month` is friendly (colleagues re-enter a code
   once a month), or keep the default 24 hours.
4. Application domain: the site address from step 2 without `https://`, e.g.
   `train-link-simulator.yourname.workers.dev`. (`wrangler.jsonc` switches off Cloudflare's per-version preview
   addresses, so this one address is the only way in.)
5. Add a policy:
   - Name: `Colleagues`, Action: **Allow**
   - Include → **Emails ending in** → `@yourcompany.com` (everyone at work), or **Emails** → list specific people
6. Login methods: keep **One-time PIN** switched on. Save.
7. Check it: open the site address in a private browser window. You should get a Cloudflare page asking for your
   email, not the "Coming soon" page.

## 4. Add the four secrets to GitHub

In the GitHub repository: **Settings** → **Secrets and variables** → **Actions** → **New repository secret**, four times:

| Name | Value |
|---|---|
| `OFCOM_API_KEY` | from your `.env` |
| `OPENCELLID_TOKEN` | from your `.env` |
| `CLOUDFLARE_API_TOKEN` | the token from step 1 (Workers Scripts → Edit) |
| `CLOUDFLARE_ACCOUNT_ID` | the Account ID from step 1 |

Secrets are write-only: nobody, including you, can read them back from GitHub, and they never reach the site.

## 5. Publish

1. Make sure the change adding `.github/workflows/publish.yml` is merged into `main`.
2. **Actions** tab → **publish** → **Run workflow** → branch `main` → **Run workflow**.
3. Wait for it to finish (about 5 hours; mostly Ofcom API calls). Open the run: its summary lists the feeds
   each route used. Any route marked as using stand-in data is still published and is retried next run.
4. Open the site address, sign in with your email and code, and share the address with your colleagues.

## Running it from then on

- It rebuilds and republishes automatically on the 17th of each month, just after the Ofcom call quota resets. To
  refresh sooner, run the workflow by hand, but see the quota note below.
- **Report downloads:** each publish builds the evidence pack (Word report + Excel appendix) for every route in the
  baseline and EDGE Rail + Fleet Connect scenarios; colleagues download them from the viewer's **Report** button.
  They add about 66 MB to the site and a few minutes to the run. A pack that fails to build does not stop the
  publish: the run shows a warning and that route's Report menu says it has no pack.
- **Adding or removing people:** edit the `Colleagues` policy in Zero Trust → Access → Applications. No rebuild needed.
- **If a run fails:** nothing is published and the previous version stays live. Open the failed run and use
  **Re-run failed jobs**. Downloads and Ofcom answers from the past week are reused, so a re-run is cheaper.
- **A "Workers Builds" check on pull requests** means the repository has been connected to Cloudflare's own Git
  builds. Delete that extra Worker in **Workers & Pages** (the publish workflow is the only thing that should deploy),
  and optionally uninstall the Cloudflare app under GitHub **Settings** → **Applications**.
- **Ofcom quota:** a full rebuild makes about 18,000 Ofcom API calls (one per postcode near the track), and the
  API key has a call quota per period; the Ofcom API answers "Out of call volume quota" once it is used up. So rebuild
  at most once per quota period. The workflow checks the quota with one call before starting, and it will not publish
  if any route's coverage fell back to stand-in data, so an exhausted quota never replaces good data on the site.
- **Company policy:** this puts work material on outside services, behind a login. If that is not allowed where you
  work, use the offline zip instead (`tcs package`, see the README).
