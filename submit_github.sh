#!/bin/bash

COMMIT_MSG="$(date '+%Y-%m-%d %H:%M:%S') update"
git add .
git commit -m "$COMMIT_MSG"
git push 