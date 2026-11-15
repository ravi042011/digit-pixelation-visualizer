# Digit Pixelation Visualizer

Draw a digit and explore its conversion to a 32 x 32 pixel grid.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Total visitors

The page shows the all-time number of browser sessions, not unique people.
Drawing, clearing, and other Streamlit reruns do not add visits. A reload,
new tab, or return visit that starts a new Streamlit session can add a visit.
Bots are not filtered. The total refreshes when the page reruns.

The counter uses PostgreSQL because local files on Streamlit Community Cloud
can disappear on restart or redeploy. Use a persistent PostgreSQL database and
configure this in the app's Streamlit Community Cloud secrets settings:

```toml
DATABASE_URL = "YOUR_REAL_POOLED_POSTGRESQL_CONNECTION_STRING"
```

Replace the placeholder with the full connection string from your provider,
not a sample URL. Keep credentials out of source control.
The previous `[visitor_counter]` / `database_url` secrets configuration still works.

### Neon local setup

This repository is linked locally to the selected Neon production branch.
Run `neon deploy` to apply `neon.ts` and write the connection strings into the
Git-ignored `.env.local`. Node.js and the dependencies in `package.json` are
only needed for Neon CLI configuration; the Streamlit app remains Python.

The Python app reads the pooled `DATABASE_URL` in this order:
the process environment, `.env.local` next to `app.py`, then Streamlit secrets.
It retains `DATABASE_URL_UNPOOLED` for administrative tools but does not use it
for normal app traffic. Existing environment values take priority.
For local development, `.streamlit/secrets.toml` is also supported and ignored.

Local `.env.local` is not uploaded to Streamlit Community Cloud. Copy its
`DATABASE_URL` value privately into the cloud app's secrets as shown above,
then save and reboot the app. Never commit this file or share its contents in chat.

The database user must be able to create the `digit_pixelation_visitors` table
and select, insert, and update its rows. The app creates the table automatically.
All instances of this app must use the same database to share the total.
Visits are incremented atomically and committed before the session is marked
as counted. The total survives app restarts as long as the database is retained.
Counting begins when this feature is configured; past visits cannot be recovered.

If configuration is missing or the database is unavailable, the page shows a
warning instead of a false count, and drawing remains available.

## Tests

```powershell
python -m unittest test_app
```

Counter tests cover session reruns, clearing, new sessions, and failures using
a mocked database boundary. Verify the live PostgreSQL connection after adding
the deployment secrets.
