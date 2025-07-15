# run.py
import os
from app import create_app

# On a platform like Render, this file is used by Gunicorn to find the 'app' object.
# The `if __name__ == '__main__':` block is for local development and is NOT run in production.
#
# --- IMPORTANT DEPLOYMENT NOTE FOR RENDER ---
# The 'WORKER TIMEOUT' and 'SIGKILL' errors in your logs indicate that the market scan
# process is taking longer than Gunicorn's default 30-second limit.
#
# To fix this, you must change the "Start Command" in your Render service settings to:
# gunicorn --workers 3 --timeout 120 -k gevent --bind 0.0.0.0:$PORT run:app
#
# This command does three things:
# 1. --workers 3: Uses a reasonable number of workers for a free tier.
# 2. --timeout 120: Increases the timeout to 120 seconds, giving the scan enough time.
# 3. -k gevent: Uses a worker type efficient for network-heavy tasks like this.
#
# Make sure to add `gevent` to your requirements.txt file (see next step).
# ----------------------------------------------------

app = create_app()

if __name__ == '__main__':
    # Use the PORT environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
