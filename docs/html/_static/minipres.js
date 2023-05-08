// Add goTo method to elements
// http://stackoverflow.com/questions/4801655/how-to-go-to-a-specific-element-on-page
(function($) {
    $.fn.goTo = function() {
        $('html, body').animate({
            scrollTop: $(this).offset().top //+ 'px'
        }, 'fast');
        return this; // for chaining...
    }
})(jQuery);

// NO good way to do this!.  Copy a hack from here
// https://stackoverflow.com/questions/901115/how-can-i-get-query-string-values-in-javascript
// https://stackoverflow.com/a/2880929
var urlParams;
(window.onpopstate = function () {
    var match,
        pl     = /\+/g,  // Regex for replacing addition symbol with a space
        search = /([^&=]+)=?([^&]*)/g,
        decode = function (s) { return decodeURIComponent(s.replace(pl, " ")); },
        query  = window.location.search.substring(1);
    urlParams = {};
    while (match = search.exec(query))
	urlParams[decode(match[1])] = decode(match[2]);
})();

// Select heading levels
var maxHeading = urlParams['h']
if (maxHeading === undefined) maxHeading = 2
var headingLevels = [];
for (h=2 ; h<maxHeading+1 ; h++)
    headingLevels.push("h"+h);
var sectionSelector = headingLevels.join(", ");

// Select heading levels which will be *hidden*
var minHeading = urlParams['minh'];
var hiddenSectionSelector = [ ];
if (minHeading) {
    for (h=Number(minHeading) ; h<7 ; h++)
	hiddenSectionSelector.push("h"+h);
}
var hiddenSectionSelector = hiddenSectionSelector.join(", ");


function section_find() {
    /* Find all the relevant sections that start each slide.  The
     * meaning of section in effect *only* matters as much as it can
     * be parsed by section_top_and_height.
     */
    //var sections = $(".title, .section");
    var sections = $(sectionSelector);
    return sections;
}

function section_top_and_height(targetSection) {
    /* Return object containging {top, height} for the passed section:
     * the offset relative to the whole page, and height of the
     * section (or zero if it can't be found).
     */
    var parent = targetSection.parentNode;

    // ReStructuredText / Sphinx sections
    if (parent.className == "section") {
	targetSection = parent;
	var top = $(targetSection).offset()["top"];
	if (targetSection.getBoundingClientRect)
	    var height = targetSection.getBoundingClientRect()["height"] || 0;
	else
	    var height = 0;
	return {"top":top, "height":height};
    }

    // Default, no fancy logic
    var top = $(targetSection).offset()["top"];
    var height = 0;
    console.log("default mode:", top, height)
    return {"top":top, "height":height};

}


function switch_slide(delta) {
    /* scroll `delta` sections forward or backwards
     */
    var sections = section_find();
    console.log(sections);

    var curPos = -10;
    // Iterate all sections until we find the last one *above* the
    // center of the screen.
    for(i=0; i<sections.length; i++) {
	screen_center = window.innerHeight/2;
	element_top = sections[i].getBoundingClientRect()["top"]
        if ( element_top < screen_center ) {
            continue;
        }
        curPos = i-1;
        break;
    }
    console.log("cur=", curPos);

    // We didn't find anything - we are at the bottom of the page.
    if (curPos == -10) {
	curPos = sections.length - 1;
    }

    // Target element we want to scroll to
    var targetPos = curPos + delta;
    console.log("target=", targetPos);

    // If we ask for -1, go directly to the top of the whole page.
    if ( targetPos == -1 ) {
	//var targetSection = $("body")
	$('html, body').animate({
            scrollTop: 0
	}, 'fast');
	return;
    }

    if ( targetPos < 0 || targetPos > (sections.length-1) ) {
    // if we would scroll past bottom, or above top, do nothing
        return;
    }

    console.log('xxxxxx');
    var targetSection = sections[targetPos];
    console.log(targetSection, typeof(targetSection));

    // Return targetSection top and height
    var secProperties = section_top_and_height(targetSection);
    var top = secProperties['top'];
    var height = secProperties['height']
    var win_height = window.innerHeight;
    //console.info(top, height, win_height)

    var scroll_to = 0;
    if (height >= win_height || height == 0) {
        scroll_to = top;
    } else {
        scroll_to = top - (win_height-height)/3.;
    }
    //console.info(top, height, win_height, scroll_to)

    $('html, body').animate({
        scrollTop: scroll_to //+ 'px'
    }, 'fast');

}


function minipres() {
    /* Enable the minipres mode:
       - call the hide() function
       - set up the scrolling listener
     */
    document.addEventListener('keydown', function (event) {
        switch(event.which) {
        case 37: // left
            switch_slide(-1);
            event.preventDefault();
            return false;
    	    break;
        //case 38: // up
        case 39: // right
            switch_slide(+1);
            event.preventDefault();
            return false;
    	    break;
        //case 40: // down
        default:
    	    return; // exit this handler for other keys
        }
    }, true)

    hide()

    // Increase space between sections
    //$("div .section").css('margin-bottom', '50%');
    $(sectionSelector).css('margin-top', '50%');

    // Reduce size/color of other sections
    if (hiddenSectionSelector.length > 0) {
        var hideNodes = $(hiddenSectionSelector);
        console.log(typeof hideNodes, hideNodes);
        for (node in hideNodes) {
            console.log("a", typeof node, node);
            node = hideNodes[node];  // what's right way to iterate values?
            console.log("b", typeof node, node);
            if (node.parentNode && node.parentNode.className == "section") {
                node = node.parentNode;
                console.log("c", typeof node, node);
                //node.css['transform'] = 'scale(.5)';
                //node.css['transform-origin'] = 'top center';
                $(node).css('color', 'lightgrey');
                //$(node).css('font-size', '20%');
                //$(node).css('visibility', 'collapse');
                //ntahousnatouhasno;
            }
        }
    }
}

function hide() {
    /* Hide all non-essential elements on the page
     */

    // This is for sphinx_rst_theme and readthedocs
    $(".wy-nav-side").remove();
    $(".wy-nav-content-wrap").css('margin-left', 0);
    $('.rst-versions').remove();  // readthedocs version selector

    // Add other formats here.
}


var slideshow = minipres;

if (window.location.search.match(/[?&](minipres|slideshow|pres)([=&]|$)/) ) {
    //minipres()
    window.addEventListener("load", minipres);
} else if (window.location.search.match(/[?&](plain)([=&]|$)/) ) {
    window.addEventListener("load", hide);
}
