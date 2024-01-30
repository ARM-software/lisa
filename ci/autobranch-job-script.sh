#! /usr/bin/env bash
# 
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2024, Arm Limited and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

function update_branch() {
	local label=$1
	local branch=$2
	local force_branch=$3

	local worktree=../${branch}-repo
	local patch=${branch}-update.patch

	lisa-combine-gitlab-mr --repo "$CI_PROJECT_PATH" --mr-label "$label" --branch "$force_branch" &&

	git fetch origin "$branch" &&

	# Work in a separate worktree so that there is no risk of folders
	# added to PATH by init_env being manipulated
	git worktree add "$worktree" --checkout "$branch" &&

	git -C "$worktree" diff --binary "HEAD..$force_branch" > "$patch" &&

	if [[ -s "$patch" ]]; then
		# Apply the patch to the index as well, so that any file created
		# is automatically added to the commit we are about to create.
		git -C "$worktree" apply --index "../origin-repo/$patch" &&
		git -C "$worktree" commit --all -m "Autocommit to $branch branch on $(date) tracking $force_branch"

		git remote set-url origin https://gitlab-ci:${GITLAB_REPO_TOKEN}@${CI_SERVER_HOST}/${CI_PROJECT_PATH}.git

		git push --force origin "$force_branch"
		git push origin "$branch"
	else
		echo "Empty patch, $branch and $force_branch branches are up to date."
	fi
}
