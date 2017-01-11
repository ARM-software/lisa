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

  config.vm.provision "shell", inline: <<-SHELL
    set -e

    if [ ! -e /home/vagrant/lisa ]; then
       ln -s /vagrant /home/vagrant/lisa
    fi

    cd /home/vagrant/lisa
    ./install_base_ubuntu.sh --install-android-sdk

    chown vagrant.vagrant /home/vagrant/lisa
    echo cd /home/vagrant/lisa >> /home/vagrant/.bashrc
    for LC in $(locale | cut -d= -f1);
    do
        echo unset $LC  >> /home/vagrant/.bashrc
    done
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
