# -*- mode: ruby -*-
# vi: set ft=ruby :


Vagrant.configure(2) do |config|
  config.vm.box = "ubuntu/trusty64"

  # Compiling pandas requires 1Gb of memory
  config.vm.provider "virtualbox" do |v|
    v.memory = 1024
  end

  # Forward ipython notebook's port to the host
  config.vm.network "forwarded_port", guest: 8888, host: 8888

  # Get VirtualBox version installed on the host
  VBOX_VERSION = `VirtualBox -help | head -1 | cut -d' ' -f5`

  config.vm.provision "shell", args: "#{VBOX_VERSION}", inline: <<-SHELL
    sudo apt-get update
    sudo apt-get install -y autoconf automake build-essential expect git \
        libfreetype6-dev libpng12-dev libtool nmap openjdk-7-jdk \
        openjdk-7-jre pkg-config python-all-dev python-matplotlib \
        python-nose python-numpy python-pip python-zmq sshpass trace-cmd \
        tree wget
    sudo pip install ipython[notebook] pandas psutil wrapt
    sudo apt-get remove -y w3m

    ln -s /vagrant /home/vagrant/lisa

    cd /home/vagrant/lisa
    ANDROID_SDK_URL="https://dl.google.com/android/android-sdk_r24.4.1-linux.tgz"
    if [ ! -e ./tools/android-sdk-linux ]; then
        echo "Downloading Android SDK [$ANDROID_SDK_URL]..."
        wget -qO- $ANDROID_SDK_URL | tar xz -C tools
        expect -c '
            set timeout -1;
            spawn ./tools/android-sdk-linux/tools/android update sdk --no-ui \
                -t tool,platform-tool,platform,build-tools-24.0.1;
            expect {
                "Do you accept the license" { exp_send "y\r" ; exp_continue }
                eof
            }
        '
    fi

    chown vagrant.vagrant /home/vagrant/lisa
    echo cd /home/vagrant/lisa >> /home/vagrant/.bashrc
    for LC in $(locale | cut -d= -f1);
    do
        echo unset $LC  >> /home/vagrant/.bashrc
    done

    # Install VirtualBox guest additions
    cd /tmp
    VBOX_VERSION=$1
    VBOX_ADDITIONS_URL="http://download.virtualbox.org/virtualbox/VER/VBoxGuestAdditions_VER.iso"
    sudo wget -q -c ${VBOX_ADDITIONS_URL//VER/$VBOX_VERSION} -O VBoxGuestAdditions_$VBOX_VERSION.iso
    sudo mount VBoxGuestAdditions_$VBOX_VERSION.iso -o loop /mnt
    expect -c '
        set timeout -1;
        spawn sudo sh /mnt/VBoxLinuxAdditions.run --nox11;
        expect {
            "Do you wish to continue" { exp_send "yes\r" ; exp_continue }
            eof
        }
    '
    sudo umount /mnt
    sudo rm *.iso
    sudo /etc/init.d/vboxadd setup
    cd -

    echo "export ANDROID_HOME=/vagrant/tools/android-sdk-linux" >> /home/vagrant/.bashrc
    echo 'export PATH=\$ANDROID_HOME/platform-tools:\$ANDROID_HOME/tools:\$PATH' >> /home/vagrant/.bashrc
    echo source init_env >> /home/vagrant/.bashrc

    echo "Virtual Machine Installation completed successfully!                "
    echo "                                                                    "
    echo "You can now access and use the virtual machine by running:          "
    echo "                                                                    "
    echo "    $ vagrant ssh                                                   "
    echo "                                                                    "
    echo "NOTE: if you exit, the virtual machine is still running. To shut it "
    echo "      down, please run:                                             "
    echo "                                                                    "
    echo "    $ vagrant suspend                                               "
    echo "                                                                    "
  SHELL
end
