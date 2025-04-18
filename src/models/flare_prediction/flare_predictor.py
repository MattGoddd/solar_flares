from sunpy.net import Fido, attrs as a
from datetime import datetime, timedelta

# Define the current time
now = datetime.now()

# Define a time window (e.g., the last 24 hours)
start_time = now - timedelta(days=1)
end_time = now

print(start_time, type(start_time), end_time, type(end_time))

# Perform the search
result = Fido.search(
    a.Time(start_time, end_time),
    a.jsoc.Series("hmi.M_720s"),
    a.jsoc.Notify("mattgoh2004@gmail.com")
)

# Display the results
print(result)
