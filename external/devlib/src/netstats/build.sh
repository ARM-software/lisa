set -e
THIS_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
pushd $THIS_DIR
ant release
jarsigner -verbose -keystore ~/.android/debug.keystore -storepass android -keypass android $THIS_DIR/bin/netstats-*.apk androiddebugkey
cp $THIS_DIR/bin/netstats-*.apk $THIS_DIR/../../devlib/instrument/netstats/netstats.apk
ant clean
popd
