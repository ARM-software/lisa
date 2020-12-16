# -*- mode: ruby -*-
# vi: set ft=ruby :


Vagrant.configure(2) do |config|
  config.vm.box = "ubuntu/bionic64"

  # Allow using tools like kernelshark
  config.ssh.forward_x11 = true

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
    # Install required packages
    ./install_base.sh --install-all

    chown -R vagrant.vagrant /home/vagrant/lisa

    # Let' use a venv local to vagrant so that we don't pollute the host one.
    # This allows to use LISA both from the host and the VM.
    export LISA_VENV_PATH=/home/vagrant/venv

    # .bashrc setup
    echo "cd /home/vagrant/lisa" >> /home/vagrant/.bashrc
    for LC in $(locale | cut -d= -f1);
    do
        echo unset $LC  >> /home/vagrant/.bashrc
    done
    echo "export LISA_VENV_PATH=$LISA_VENV_PATH" >> /home/vagrant/.bashrc
    echo 'source init_env' >> /home/vagrant/.bashrc

    # Trigger the creation of a venv and check that everything works well
    if ! su vagrant bash -c 'tools/tests.sh'; then
      echo "Self tests FAILED !"
    else
      echo "Virtual Machine Installation completed successfully!                "
    fi

    echo "You can now access and use the virtual machine by running:          "
    echo "                                                                    "
    echo "    $ vagrant ssh                                                   "
    echo "                                                                    "
    echo "NOTE: if you exit, the virtual machine is still running. To shut it "
    echo "      down, please run:                                             "
    echo "                                                                    "
    echo "    $ vagrant suspend                                               "
    echo "                                                                    "
    echo " To destroy it, use:                                                "
    echo "                                                                    "
    echo "    $ vagrant destroy                                               "
    echo "                                                                    "
  SHELL
end
