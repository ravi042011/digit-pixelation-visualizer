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
[visitor_counter]
database_url = "postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require"
```

Replace the placeholders with your database connection details. Keep credentials
out of source control. For local development, put the same configuration in
`.streamlit/secrets.toml`, which is already ignored by Git.

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
