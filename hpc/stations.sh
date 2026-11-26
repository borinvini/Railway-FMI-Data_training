# The target stations of the multi-station grid. Sourced, never executed.
#
# Consumers:
#   hpc/stage_data.sh --stations      what goes to /scratch/.../railway-fmi/stations/<ST>/
#   hpc/submit-stations.sh            one array of the scenario grid per station
#   hpc/fetch-stations.sh             what comes back, and into which archive folder
#   scripts/preprocess_stations.py    reads this line through scripts/stations.py
#
# ORDER IS PART OF THE CONTRACT. A station's position + 1 is its number in the
# results archive ("AI Results/<BINARY|NUMERIC>/<N> - <ST>"), so reordering this
# list files a station's results under another station's number. Append only.
#
# Keep it on one line: scripts/stations.py parses exactly this assignment.
STATIONS=(HKI PSL TPE SK YV OL ROI)
