#/!bin/sh

SYSFS_BASE="/sys/devices/12c60000.i2c/i2c-4/i2c-dev/i2c-4/device/"
SYSFS_ARM=$SYSFS_BASE"/4-0040"
SYSFS_KFC=$SYSFS_BASE"/4-0045"

if [ $# -lt 2 ]; then
    echo "Usage: $0 samples period_s [arm|kfc]"
    exit 1
fi

SAMPLES=$1
PERIOD=$2
DEVICE=${3:-"arm"}

case $DEVICE in
"arm")
    SYSFS_ENABLE=$SYSFS_ARM"/enable"
    SYSFS_W=$SYSFS_ARM"/sensor_W"
    ;;
"kfc")
    SYSFS_ENABLE=$SYSFS_KFC"/enable"
    SYSFS_W=$SYSFS_KFC"/sensor_W"
    ;;
esac

echo "Samping $SAMPLES time, every $PERIOD [s]:"
echo "   $SYSFS_W"

rm samples_w.txt 2>/dev/null
echo 1 > $SYSFS_ENABLE
sleep 1

while [ 1 ]; do
    sleep $PERIOD
    cat $SYSFS_W >> samples_w.txt
    SAMPLES=$((SAMPLES-1))
    [ $SAMPLES -eq 0 ] && break
done

echo 0 > $SYSFS_ENABLE
