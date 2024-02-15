#! /bin/bash


# Install source location for a git tree
# $1: name of the project
# $2: path to the cloned git repo
# $3: relative path in repo of the COPYING licence file
install_readme() {
    name=$1
    repo=$2
    licence_file="$repo/$3"
    src_info="$LISA_ARCH_ASSETS/README.$name"

    sha1=$(git -C "$repo" rev-list -1 HEAD)
    remote=$(git -C "$repo" config --get remote.origin.url)
    printf "Sources of %s available at:\nGit commit: %s\nGit repository: %s\n" "$name" "$sha1" "$remote" > "$src_info"

    printf "\n\n%s\n\n" "Build host info:" >> "$src_info"
    cat /etc/os-release >> "$src_info"

    printf "\n\n%s\n\n" "Build recipe:" >> "$src_info"

    # Dump the env var that are directly used by the recipe
    local used_env=($LISA_ASSET_RECIPE_USED_ENV)
    for var in ${used_env[@]}; do
        val=${!var}
        # remove LISA_HOME value if that is a prefix in the value, to
        # avoid leaking local paths
        val=${val#"$LISA_HOME"}
        printf "export %s=%q\n" "$var" "$val" >> "$src_info"
    done

    cat "$LISA_ASSET_RECIPE" >> "$src_info"

    printf "\n\nThe sources were distributed under the following licence (content of %s):\n\n" "$licence_file" >> "$src_info"
    cat "$licence_file" >> "$src_info"

    # If compiled with musl-libc, add the COPYRIGHT
    if [[ $USE_MUSL_LIB == 1 ]]
    then
        printf "\n\nThe sources were compiled with musl-libc (content of COPYRIGHT):\n\n" >> "$src_info"

        # For some reason wget -O - >> "$src_info" does not work. This may be
        # due to bugs in the VirtualBox driver, when using a chroot inside a
        # Vagrant VM.
        musl_license=$(mktemp)
        wget https://git.musl-libc.org/cgit/musl/plain/COPYRIGHT -O "$musl_license"
        cat "$musl_license" >> "$src_info"
        rm "$musl_license"
    fi
}

# Copy files tracked by git only
# $1: source folder with files to copy.
# $2: destination folders to copy the content of $1 into.
copy_from_git() {
    local src=$1
    local dst=$(readlink -f "$2")
    local src_base=$(basename "$src")

    (
        cd "$src"
        set -o noglob
        IFS=$'\n' local files=($(git ls-files "$src"))
        set +o noglob

        for f in "${files[@]}"; do
            local _dst="$dst/$src_base/$f"
            mkdir -p "$(dirname $_dst)" &&
            cp -v "$f" "$_dst"
        done
    )
}
