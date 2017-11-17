#    Copyright 2014-2016 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from wa import ApkUiautoWorkload, Parameter


class Googleplaybooks(ApkUiautoWorkload):

    name = 'googleplaybooks'
    package_names = ['com.google.android.apps.books']

    description = '''
    A workload to perform standard productivity tasks with googleplaybooks.
    This workload performs various tasks, such as searching for a book title
    online, browsing through a book, adding and removing notes, word searching,
    and querying information about the book.

    Test description:
    1. Open Google Play Books application
    2. Dismisses sync operation (if applicable)
    3. Searches for a book title
    4. Adds books to library if not already present
    5. Opens 'My Library' contents
    6. Opens selected book
    7. Gestures are performed to swipe between pages and pinch zoom in and out of a page
    8. Selects a specified chapter based on page number from the navigation view
    9. Selects a word in the centre of screen and adds a test note to the page
    10. Removes the test note from the page (clean up)
    11. Searches for the number of occurrences of a common word throughout the book
    12. Switches page styles from 'Day' to 'Night' to 'Sepia' and back to 'Day'
    13. Uses the 'About this book' facility on the currently selected book

    NOTE: This workload requires a network connection (ideally, wifi) to run,
          a Google account to be setup on the device, and payment details for the account.
          Free books require payment details to have been setup otherwise it fails.
          Tip: Install the 'Google Opinion Rewards' app to bypass the need to enter valid
          card/bank detail.

    Known working APK version: 3.15.5
    '''

    parameters = [
        Parameter('search_book_title', kind=str, default='Nikola Tesla: Imagination and the Man That Invented the 20th Century',
                  description="""
                  The book title to search for within Google Play Books archive.
                  The book must either be already in the account's library, or free to purchase.
                  """),
        Parameter('library_book_title', kind=str, default='Nikola Tesla',
                  description="""
                  The book title to search for within My Library.
                  The Library name can differ (usually shorter) to the Store name.
                  If left blank, the ``search_book_title`` will be used.
                  """),
        Parameter('select_chapter_page_number', kind=int, default=4,
                  description="""
                  The Page Number to search for within a selected book's Chapter list.
                  Note: Accepts integers only.
                  """),
        Parameter('search_word', kind=str, default='the',
                  description="""
                  The word to search for within a selected book.
                  Note: Accepts single words only.
                  """),
        Parameter('account', kind=str, mandatory=False,
                  description="""
                  If you are running this workload on a device which has more than one
                  Google account setup, then this parameter is used to select which account
                  to select when prompted.
                  The account requires the book to have already been purchased or payment details
                  already associated with the account.
                  If omitted, the first account in the list will be selected if prompted.
                  """),
    ]

    # This workload relies on the internet so check that there is a working
    # internet connection
    requires_network = True

    def init_resources(self, context):
        super(Googleplaybooks, self).init_resources(context)
        self.gui.uiauto_params['search_book_title'] = self.search_book_title
        # If library_book_title is blank, set it to the same as search_book_title
        if not self.library_book_title:  # pylint: disable=access-member-before-definition
            self.library_book_title = self.search_book_title  # pylint: disable=attribute-defined-outside-init
        self.gui.uiauto_params['library_book_title'] = self.library_book_title
        self.gui.uiauto_params['chapter_page_number'] = self.select_chapter_page_number
        self.gui.uiauto_params['search_word'] = self.search_word
        self.gui.uiauto_params['account'] = self.account
