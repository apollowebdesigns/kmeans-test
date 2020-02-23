#!/bin/bash
HOST="localhost"
USER="anonymous"
PASS="twisted@"
LDIR="./"    # can be empty
RDIR=""   # can be empty

wget ftp://$HOST/$RDIR \
  --continue \               # resume on files that have already been partially transmitted
  --mirror \                 # --recursive --level=inf --timestamping --no-remove-listing
  --no-host-directories \    # don't create 'ftp://src/' folder structure for synced files
  --ftp-user=$USER \
  --ftp-password=$PASS \
