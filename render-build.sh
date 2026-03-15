#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system dependencies for 'av' and 'opencv'
# Render's Ubuntu environment allows apt-get in the build step
apt-get update && apt-get install -y \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libgl1

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt