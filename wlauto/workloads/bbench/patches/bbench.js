//Author: Anthony Gutierrez

var bb_site = [];
var bb_results = [];
var globalSiteIndex = 0;
var numWebsites = 9;
var bb_path = document.location.pathname;
var bb_home = "file:///" + bb_path.substr(1, bb_path.lastIndexOf("bbench") + 5);
var num_iters = 0;
var init = false;

function generateSiteArray(numTimesToExecute) {
    for (i = 0; i < numTimesToExecute * numWebsites; i += numWebsites) {
        bb_site[i+0] = bb_home + "/sites/amazon/www.amazon.com/index.html";
        bb_site[i+1] = bb_home + "/sites/bbc/www.bbc.co.uk/index.html";
        bb_site[i+2] = bb_home + "/sites/cnn/www.cnn.com/index.html";
        bb_site[i+3] = bb_home + "/sites/craigslist/newyork.craigslist.org/index.html";
        bb_site[i+4] = bb_home + "/sites/ebay/www.ebay.com/index.html";
        bb_site[i+5] = bb_home + "/sites/google/www.google.com/index.html";
//        bb_site[i+6] = bb_home + "/sites/youtube/www.youtube.com/index.html";
        bb_site[i+6] = bb_home + "/sites/msn/www.msn.com/index.html";
        bb_site[i+7] = bb_home + "/sites/slashdot/slashdot.org/index.html";
        bb_site[i+8] = bb_home + "/sites/twitter/twitter.com/index.html";
//        bb_site[i+10] = bb_home + "/sites/espn/espn.go.com/index.html";
    }

    bb_site[i] = bb_home + "/results.html";
}


/* gets the URL parameters and removes from window href */
function getAndRemoveURLParams(windowURL, param) {
    var regex_string = "(.*)(\\?)" + param + "(=)([0-9]+)(&)(.*)";
    var regex = new RegExp(regex_string);
    var results = regex.exec(windowURL.value);

    if (results == null)
        return "";
    else {
        windowURL.value = results[1] + results[6];
        return results[4];
    }
}

/* gets the URL parameters */
function getURLParams(param) {
    var regex_string = "(.*)(\\?)" + param + "(=)([0-9]+)(&)(.*)";
    var regex = new RegExp(regex_string);
    var results = regex.exec(window.location.href);

    if (results == null)
        return "";
    else
        return results[4];
}

/* gets all the parameters */
function getAllParams() {
    var regex_string = "(\\?.*)(\\?siteIndex=)([0-9]+)(&)";
    var regex = new RegExp(regex_string);
    var results = regex.exec(window.location.href);
    /*alert(" Result is 1: " + results[1] + " 2: " + results[2] + " 3: " + results[3]);*/

    if (results == null)
        return "";
    else
        return results[1];
}

/* sets a cookie */
function setCookie(c_name, value) {
    var c_value = escape(value) + ";";
    document.cookie = c_name + "=" + c_value + " path=/";
}

/* gets a cookie */
function getCookie(c_name) {
    var cookies = document.cookie.split(";");
    var i, x, y;

    for (i = 0; i < cookies.length; ++i) {
        x = cookies[i].substr(0, cookies[i].indexOf("="));
        y = cookies[i].substr(cookies[i].indexOf("=") + 1);
        x = x.replace(/^\s+|\s+$/g,"");

        if (x == c_name)
            return unescape(y);
    }
}

/* start the test, simply go to site 1. */
function startTest(n, del, y) {
    //var start_time = (new Date()).getTime();
    //setCookie("PreviousTime", start_time);

    init = true;

    generateSiteArray(n);
    siteTest(bb_site[0], globalSiteIndex, new Date().getTime(), "scrollSize=" + y + "&?scrollDelay=" + del + "&?iterations=" + n + "&?" + "StartPage");
    //siteTest(bb_site[0], globalSiteIndex, new Date().getTime(), "scrollDelay=" + del + "&?iterations=" + n + "&?" + "StartPage");
    //goToSite(bb_site[0], new Date().getTime());
}

/* jump to the next site */
function goToSite(site) {
    curr_time = new Date().getTime();
    setCookie("CGTPreviousTime", curr_time);
    site+="?CGTPreviousTime="+curr_time+"&";
    window.location.href = site;
}

/*
  the test we want to run on the site.
  for now, simply scroll to the bottom
  and jump to the next site. in the
  future we will want to do some more
  realistic browsing tests.
*/
function siteTest(nextSite, siteIndex, startTime, siteName) {
    if (!init) {
        var iterations = getURLParams("iterations");
        var params = getAllParams();
        var delay = getURLParams("scrollDelay");
        var verticalScroll = getURLParams("scrollSize");
        generateSiteArray(iterations);
        nextSite = bb_site[siteIndex] + params;
    }
    else {
        var delay = 500;
        var verticalScroll = 500;
    }
    var cgtPreviousTime = getURLParams("CGTPreviousTime");
    var load_time = 0;
    siteIndex++;
    if (siteIndex > 1) {
       cur_time = new Date().getTime();
//       alert("previous " + cgtPreviousTime + " foo " + getCookie("CGTPreviousTime"));
       load_time = (cur_time - cgtPreviousTime);
       setCookie("CGTLoadTime", load_time);
//       diff = cur_time-startTime;
//       alert("starttime "+startTime+" currtime "+ cur_time + " diff " + diff + "load_time " + load_time );
    }
    setTimeout(function() {
        scrollToBottom(0, verticalScroll, delay,load_time,
        function(load_time_param){
            cur_time = new Date().getTime();
            load_time = (cur_time - startTime);
            //load_time = (cur_time - getCookie("PreviousTime"));
            // alert("Done with this site! " + window.cur_time + " " + startTime + " " + window.load_time);
            //alert("Done with this site! " + window.cur_time + " " + getCookie("PreviousTime") + " " + window.load_time);
            //goToSite(nextSite + "?iterations=" + iterations  + "&?" + siteName + "=" + load_time + "&" + "?siteIndex=" + siteIndex + "&" );
//            alert("loadtime in cookie="+ getCookie("CGTLoadTime")+" loadtime in var="+load_time_param);
            goToSite(nextSite + "?" + siteName + "=" + load_time_param + "&" + "?siteIndex=" + siteIndex + "&" );
        }
    );},(siteIndex > 1) ? 1000 : 0);
}

/*
  scroll to the bottom of the page in
  num_y pixel increments. may want to
  do some horizontal scrolling in the
  future as well.
*/
function scrollToBottom(num_x, num_y, del, load_time, k) {
    ++num_iters;
    var diff = document.body.scrollHeight - num_y * num_iters;
    //var num_scrolls = 0;

    if (diff > num_y) {
            //self.scrollBy(num_x, num_y);
            //setTimeout(function(){self.scrollBy(num_x, num_y); /*diff -= 100;*/ scrollToBottom(num_x, num_y, k);}, 2);
            setTimeout(function(){self.scrollBy(num_x, num_y); /*diff -= 100;*/ scrollToBottom(num_x, num_y, del, load_time,k);}, del);
    }
    else{
	k(load_time);
    }
}
