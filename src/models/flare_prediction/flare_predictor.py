from sunpy.net import Fido, attrs as a

query = Fido.search(
    a.Time("2025-04-04 12:00", "2025-04-04 12:12"),
    a.jsoc.Series("hmi.M_720s"),
    a.jsoc.Notify("test@example.com")
)

print(query)