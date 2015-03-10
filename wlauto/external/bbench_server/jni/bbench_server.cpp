/*    Copyright 2012-2015 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

/**************************************************************************/
/* Simple HTTP server program that will return on accepting connection    */
/**************************************************************************/

/* Tested on Android ICS browser and FireFox browser */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>

#define SERVERPORT "3030"

void ExitOnError(int condition, const char *msg)
{
   if(condition) { printf("Server: %s\n", msg); exit(1);}
}

void *GetInetAddr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET)
    {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }
    else
    {
	    return &(((struct sockaddr_in6*)sa)->sin6_addr);
	}
}

int main(int argc, char *argv[])
{

    socklen_t addr_size;
    struct addrinfo hints, *res;
    int server_fd, client_fd;
    int retval;
    int timeout_in_seconds;

    // Get the timeout value in seconds
    if(argc < 2)
    {
        printf("Usage %s <timeout in seconds>\n", argv[0]);
        exit(1);
    }
    else
    {
        timeout_in_seconds = atoi(argv[1]);
        printf("Server: Waiting for connection on port %s with timeout of %d seconds\n", SERVERPORT, timeout_in_seconds);

    }

	/**************************************************************************/
	/* Listen to a socket	                                                  */
	/**************************************************************************/
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;  // use IPv4 or IPv6, whichever
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;     // fill in my IP for me

    getaddrinfo(NULL, SERVERPORT, &hints, &res);


    server_fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    ExitOnError(server_fd < 0, "Socket creation failed");

    retval = bind(server_fd, res->ai_addr, res->ai_addrlen);
    ExitOnError(retval < 0, "Bind failed");

    retval = listen(server_fd, 10);
    ExitOnError(retval < 0, "Listen failed");

	/**************************************************************************/
	/* Wait for connection to arrive or time out							  */
	/**************************************************************************/
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(server_fd, &readfds);

    // Timeout parameter
    timeval tv;
    tv.tv_sec  = timeout_in_seconds;
    tv.tv_usec = 0;

    int ret = select(server_fd+1, &readfds, NULL, NULL, &tv);
    ExitOnError(ret <= 0, "No connection established, timed out");
	ExitOnError(FD_ISSET(server_fd, &readfds) == 0, "Error occured in select");

	/**************************************************************************/
	/* Accept connection and print the information							  */
	/**************************************************************************/
    {
		struct sockaddr_storage client_addr;
		char client_addr_string[INET6_ADDRSTRLEN];
    	addr_size = sizeof client_addr;
    	client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &addr_size);
    	ExitOnError(client_fd < 0, "Accept failed");

    	inet_ntop(client_addr.ss_family,
    			  GetInetAddr((struct sockaddr *)&client_addr),
    			  client_addr_string,
    			  sizeof client_addr_string);
    	printf("Server: Received connection from %s\n", client_addr_string);
	}


    /**************************************************************************/
    /* Send a acceptable HTTP response									      */
    /**************************************************************************/
    {

		char response[] = "HTTP/1.1 200 OK\r\n"
                          "Content-Type: text/html\r\n"
                          "Connection: close\r\n"
                          "\r\n"
                          "<html>"
                          "<head>Local Server: Connection Accepted</head>"
                          "<body></body>"
                          "</html>";
		int  bytes_sent;
        bytes_sent = send(client_fd, response, strlen(response), 0);
		ExitOnError(bytes_sent < 0, "Sending Response failed");
    }


    close(client_fd);
    close(server_fd);
    return 0;
}
