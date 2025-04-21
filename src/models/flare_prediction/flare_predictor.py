from sunpy.net import Fido, attrs as a
from datetime import datetime, timedelta, timezone

# Define the current time
now = datetime.now()

# Define a time window (e.g., the last 24 hours)
start_time = now - timedelta(days=1)
end_time = now

now = datetime.now(timezone.utc)
test = now - timedelta(hours=1)

print(start_time, type(start_time), end_time, type(end_time))

print("Starting download for file 'hmi.sharp_720s_nrt'")

# Perform the search
result = Fido.search(
    a.Time(test, now),
    a.jsoc.Series("hmi.B_720s_nrt"),
    a.jsoc.Notify("mattgoh2004@gmail.com")
)

# Display the results
print(result)
