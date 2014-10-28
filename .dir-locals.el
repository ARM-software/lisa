((nil . ((eval . (progn
                   (require 'projectile)
                   (puthash (projectile-project-root)
                            "nosetests --processes=-1 --process-timeout=60"
                            projectile-test-cmd-map))))))
