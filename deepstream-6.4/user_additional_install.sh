# additional components the user can self install
apt-get update
apt-get install -y gstreamer1.0-libav
# ubuntu 22.04
apt-get install --reinstall -y gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly  libswresample-dev libavutil-dev libavutil56 libavcodec-dev libavcodec58 libavformat-dev libavformat58 libavfilter7 libde265-dev libde265-0 libx265-199 libx264-163 libvpx7 libmpeg2encpp-2.1-0 libmpeg2-4 libmpg123-0
echo "Deleting GStreamer cache"
rm -rf ~/.cache/gstreamer-1.0/
