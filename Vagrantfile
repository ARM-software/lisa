# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |config|
  config.vm.box = "ubuntu/focal64"

  # Allow using tools like kernelshark
  config.ssh.forward_x11 = true

  # Compiling pandas requires 1Gb of memory
  config.vm.provider "virtualbox" do |v|
    v.memory = 1024
  end

  # Forward ipython notebook's port to the host
  if !ENV['VAGRANT_FORWARD_JUPYTER'] == '0'
    config.vm.network "forwarded_port", guest: 8888, host: 8888
  end

  config.vm.provision "shell", inline: <<-SHELL
    set -e

    # Workaround the Virtualbox issue with synced folder file renaming and
    # deletion
    # https://github.com/hashicorp/vagrant/issues/12057
    # and https://www.virtualbox.org/ticket/8761
    #
    # Note: changes in the lowerdir by the host may result in unexpected
    # behaviors in the guest, but it should not crash or corrupt data.
    lowerdir='/vagrant/external/'
    upperdir='/home/vagrant/lisa-external-upper'
    workdir='/home/vagrant/lisa-external-work'

    mkdir -p "$upperdir"
    mkdir -p "$workdir"
    echo "overlay $lowerdir overlay lowerdir=$lowerdir,upperdir=$upperdir,workdir=$workdir" >> /etc/fstab

    # Apply the changes to fstab
    sudo systemctl daemon-reload
    sudo systemctl restart local-fs.target

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
    echo "export LISA_VENV_PATH=$LISA_VENV_PATH" >> /home/vagrant/activate-lisa
    echo 'source init_env' >> /home/vagrant/activate-lisa
    echo 'source /home/vagrant/activate-lisa' >> /home/vagrant/.bashrc

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
