#!/usr/bin/env bash
lsof -ti tcp:8000 | xargs kill -9 2>/dev/null
lsof -ti tcp:5500 | xargs kill -9 2>/dev/null
echo "Stopped anything on 8000/5500."
