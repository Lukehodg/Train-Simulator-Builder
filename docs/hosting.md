# Hosting the simulator for colleagues

The viewer is a static website: the live feeds are fetched by the pipeline when the routes are built, not while
someone is using it. So hosting it "with live feeds" means two pieces:

- **GitHub Actions** (`.github/workflows/publish.yml`) rebuilds all 18 routes from the live feeds on the 1st of each
  month, or whenever you run it by hand, then publishes the result. Your API keys live only there, as GitHub secrets.
  GitHub's machines have open internet, so OpenStreetMap works and every route gets its exact track line.
- **Cloudflare Pages** serves the site, and **Cloudflare Access** decides who can open it. Colleagues open the link,
  enter their work email and type the one-time code Cloudflare emails them. No company sign-in integration is needed.

Everything below is free: Cloudflare Access for up to 50 users, and a monthly rebuild (about 2-3.5 hours) fits
within GitHub's free Actions minutes for a private repository. Setup takes about 20 minutes. Cloudflare changes its
dashboard wording from time to time, so a label may differ slightly from what is written here.

> Do the steps in this order. The site must be locked (step 3) before the first real publish (step 5).

## 1. Create a Cloudflare account and note two values

1. Sign up at <https://dash.cloudflare.com/sign-up> (free plan).
2. **Account ID:** in the dashboard, open **Workers & Pages**; the Account ID is in the right-hand column
   (or use the **...** menu next to your account name and choose **Copy account ID**). Keep it for step 4.
3. **API token:** click your profile icon (top right) → **My Profile** → **API Tokens** → **Create Token** →
   **Create Custom Token**.
   - Token name: `Train Link Simulator publish`
   - Permissions: **Account** → **Cloudflare Pages** → **Edit**
   - Account Resources: **Include** → your account
   - **Continue to summary** → **Create Token**, then copy the token. It is shown only once; keep it for step 4.

## 2. Create the (empty) site to get its address

1. **Workers & Pages** → **Create** → **Pages** → **Upload assets** (direct upload).
2. Project name: `train-link-simulator`. This must match `PAGES_PROJECT` in the workflow; if you choose another
   name, change it there too.
3. Upload a placeholder: on your computer make a folder containing one file called `index.html` with the text
   `Coming soon`, drag the folder in and click **Deploy site**.
4. Note the address Cloudflare gives the site, for example `https://train-link-simulator.pages.dev`. If that name
   was taken, Cloudflare adds a suffix; use whatever address it shows.

## 3. Lock the site to your colleagues (Cloudflare Access)

1. In the dashboard sidebar open **Zero Trust**. The first time, pick a team name (anything, e.g. `yourname-tools`)
   and the **Free** plan. It may ask for a card even on the free plan; you are not charged for up to 50 users.
2. **Access** → **Applications** → **Add an application** → **Self-hosted**.
3. Application name: `Train Link Simulator`. Session duration: `1 month` is friendly (colleagues re-enter a code
   once a month), or keep the default 24 hours.
4. Application domain: the site address from step 2 without `https://`, e.g. `train-link-simulator.pages.dev`.
   Add a second domain `*.train-link-simulator.pages.dev` so preview copies of the site are locked too.
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
| `CLOUDFLARE_API_TOKEN` | the token from step 1 |
| `CLOUDFLARE_ACCOUNT_ID` | the Account ID from step 1 |

Secrets are write-only: nobody, including you, can read them back from GitHub, and they never reach the site.

## 5. Publish

1. Make sure the change adding `.github/workflows/publish.yml` is merged into `main`.
2. **Actions** tab → **publish** → **Run workflow** → branch `main` → **Run workflow**.
3. Wait for it to finish (about 2-3.5 hours; mostly Ofcom API calls). Open the run: its summary lists the feeds
   each route used. Any route marked as using stand-in data is still published and is retried next run.
4. Open the site address, sign in with your email and code, and share the address with your colleagues.

## Running it from then on

- It rebuilds and republishes automatically on the 1st of each month. To refresh sooner, run the workflow by hand.
- **Adding or removing people:** edit the `Colleagues` policy in Zero Trust → Access → Applications. No rebuild needed.
- **If a run fails:** nothing is published and the previous version stays live. Open the failed run and use
  **Re-run failed jobs**. Downloads and Ofcom answers from the past week are reused, so a re-run is cheaper.
- **Ofcom quota:** each full rebuild makes a few thousand Ofcom API calls (one per postcode near the track). Monthly
  is comfortable; running it many times a day may hit your API product's limits.
- **Company policy:** this puts work material on outside services, behind a login. If that is not allowed where you
  work, use the offline zip instead (`tcs package`, see the README).
