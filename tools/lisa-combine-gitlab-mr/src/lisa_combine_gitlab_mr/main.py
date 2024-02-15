#! /usr/bin/env python3
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

import subprocess
from itertools import chain
from tempfile import NamedTemporaryFile
import json
from collections import ChainMap
from operator import itemgetter
import argparse
import logging

import requests

# Use HTTP APIs of GitLab to retrieve associated mrs
# of this project.
def get_gitlab_mrs(api_url, project_id, api_token=None, state="opened", scope="all", labels=None):
    if labels:
        labels = ','.join(labels)
        labels = f'&labels={labels}'
    else:
        labels = ''

    def call_gitlab_api(endpoint):
        headers = {
            k: v
            for k, v in [('PRIVATE-TOKEN', api_token)]
            if v
        }
        r = requests.get(f'{api_url}/{endpoint}', headers=headers)
        r.raise_for_status()
        return r

    # Get the clone URL of the original repo
    clone_url = call_gitlab_api(
        f"projects/{project_id}"
    ).json()['http_url_to_repo']

    def get_mr(mr):
        iid = mr['iid']
        mr_commit_response = call_gitlab_api(
            f"projects/{project_id}/merge_requests/{iid}/commits"
        )
        mr_commit_response = mr_commit_response.json()
        assert isinstance(mr_commit_response, list)
        commits_count = len(mr_commit_response)

        return dict(
            sha=mr['sha'],
            # We use the magic branch associated with the merge-request instead
            # of the "real" source branch and its associated repo, as the
            # latter could be a private fork we don't have direct access to.
            source_branch=f'refs/merge-requests/{iid}/head',
            commits_count=commits_count,
            clone_url=clone_url,
        )

    results = []
    # obvioulsy at start 1
    page_number = 1
    while True:
        mr_response = call_gitlab_api(
            f"merge_requests?state={state}&scope={scope}{labels}&page={page_number}"
        )
        results.extend(
            get_mr(mr)
            for mr in mr_response.json()
            if mr.get("project_id") == project_id
        )

        # handle paging - only required for mr
        if not mr_response.headers["X-Next-Page"]:
            break
        page_number = mr_response.headers["X-Next-Page"]

    return results

def main():
    parser = argparse.ArgumentParser(
        description="""
        Combine gitlab merge requests with the given tag into a branch, rebasing all
        MRs on top of each other.
        """,
    )

    parser.add_argument('--server', required=True, help='Gitlab server URL')
    parser.add_argument('--api-url', required=True, help='Gitlab API URL')
    parser.add_argument('--api-token', help='Gitlab API token. If omitted, anonymous requests will be used which may fail')
    parser.add_argument('--project-id', type=int, required=True, help='Gitlab Project ID')
    parser.add_argument('--repo', required=True, help='Gitlab repository as owner/name')
    parser.add_argument('--mr-label', action='append', required=True, help='Merge request labels to look for')
    parser.add_argument('--branch', required=True, help='Name of the branch to be created. If the branch exists, it will be forcefully updated')

    args = parser.parse_args()

    owner, repo = args.repo.split('/', 1)
    labels = args.mr_label
    branch = args.branch
    server = args.server
    api_url = args.api_url
    api_token = args.api_token
    project_id = args.project_id

    logging.basicConfig(level=logging.INFO)

    def make_topic(mr):
        remote = f'remote_{mr["sha"]}'
        return (
            {
                remote: {
                    'url': mr["clone_url"]
                }
            },
            {
                'name': mr["source_branch"],
                'remote': remote,
                'nr-commits': mr["commits_count"],
                'tip': mr["source_branch"],
            }
        )

    topics = [
        make_topic(mr)
        for mr in get_gitlab_mrs(
            api_url=api_url,
            api_token=api_token,
            project_id=project_id,
            labels=labels
        )
    ]

    remotes, topics = zip(*topics) if topics else ([], [])
    remotes = dict(ChainMap(*chain(
        [{
            'gitlab': {
            'url': f'https://{server}/{owner}/{repo}.git'
            }
        }],
        remotes
    )))

    conf = {
        'rebase-conf': {
            'rr-cache': './rr-cache',
            'remotes': remotes,
            'base': {
                'remote': 'gitlab',
                'ref': 'main',
            },
            'topics': sorted(topics, key=itemgetter('name'))
        }
    }
    conf = json.dumps(conf, indent=4)
    logging.info(conf)

    with NamedTemporaryFile(mode='w+', suffix='.manifest.json') as f:
        f.write(conf)
        f.flush()

        manifest = f.name

        cmd = ['batch-rebase', 'create', '.', '--manifest', manifest, '--create-branch', branch]
        logging.info(f'Running {" ".join(map(str, cmd))}')
        subprocess.check_call(cmd)
