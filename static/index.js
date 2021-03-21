function defineJQueryObjects(){
    // These objects are inside the form that we rewrite using Ajax, so they
    // need to be recreated each time. This is a helper function for that.
    rule_out_agi_by_field = $('#rule_out_agi_by')
    virtual_successes_field = $('#virtual_successes')
    regime_start_year_field = $('#regime_start_year')
    first_trial_probability_field = $('#first_trial_probability')
    g_exp_field = $('#g_exp')
    g_act_field = $('#g_act')
    init_weight_calendar_field = $('#init_weight_calendar')
    init_weight_researcher_field = $('#init_weight_researcher')
    init_weight_agi_impossible_field = $('#init_weight_agi_impossible')
    relative_imp_res_comp_field = $('#relative_imp_res_comp')
    comp_spending_assumption_field = $('#comp_spending_assumption')

    init_weight_comp_relative_res_field = $('#init_weight_comp_relative_res')
    init_weight_lifetime_field = $('#init_weight_lifetime')
    init_weight_evolution_field = $('#init_weight_evolution')
}

$(document).ready(function() {
    defineJQueryObjects()

    // Loading indicator
    var loading_indicator = $('#loading_indicator').hide();
    $(document)
      .ajaxStart(function () {
        loading_indicator.show();
      })
      .ajaxStop(function () {
        loading_indicator.hide();
      });

    if (window.visualViewport) {
        function onWindowScroll() {
            loading_indicator.css('top',window.visualViewport.offsetTop + 25 + "px")
            loading_indicator.css('left',window.visualViewport.offsetLeft + 25 + "px")
        }
        onWindowScroll();
        window.visualViewport.addEventListener("resize", onWindowScroll);
        window.visualViewport.addEventListener("scroll", onWindowScroll);
        // https://developers.google.com/web/updates/2017/09/visual-viewport-api#gotchas
        window.addEventListener('scroll', onWindowScroll);
    }

    // Create tooltips
    createTooltips()


$(document).on('submit', '.main_form', function(event){
    makeAJAXCall(event)
});

})

function makeAJAXCall(event){
    // Submit the form via AJAX to avoid reloading the page
    if (event) {
        event.preventDefault();
    }

    var form = $("#main_form");
    var url = form.attr('action');

    // remember scroll position
    var scroll = $(window).scrollTop();

    $.ajax({
           type: "POST",
           url: url,
           data: form.serialize(), // serializes the form's elements.
           success: function(response)
           {
               // parses the big long string into an array of DOM nodes
               nodes = $.parseHTML(response, keepScripts=true)

               // re-writes the page
               $("#main_content").replaceWith($(nodes).filter('#main_content'))

               // scroll to any errors
               errors = $('.errors')
               if (errors.length) {
                   $('html, body').animate({
                       'scrollTop': errors.offset().top - $(document).height()*0.1
                   });
               }

           },
           error: function (response){
               // Show the user the error
               $("#main_content").html(
                   "<p class='errors'>Internal server error. <a href='javascript:location.reload()'>Reload with default values</a>.</p>"
               )
           }
         });

}

// Hack to move y-axes that were being cut off. Customizing axis placement appears to be impossible or poorly documented in mpld3
function moveAxes(){
    labels = $('.mpld3-text[transform]') // add [transform] so we only catch rotated labels
    labels.attr('transform', (i, value) => `${value || ""} translate(0 10)`)
}

$( window ).on( "load", function (){
    moveAxes()
})

$( document ).ajaxComplete(function(){
    defineJQueryObjects()
    createTooltips()
    moveAxes()
})

function fillLow() {
    rule_out_agi_by_field.val(2020)
    virtual_successes_field.val(0.5)
    regime_start_year_field.val(1956)
    first_trial_probability_field.val(0.001)
    g_exp_field.val(4.3)
    g_act_field.val(7)

    init_weight_calendar_field.val(.5)
    init_weight_researcher_field.val(.3)
    init_weight_agi_impossible_field.val(.2)
    init_weight_comp_relative_res_field.val(0)
    init_weight_lifetime_field.val(0)
    init_weight_evolution_field.val(0)

    $('#main_form').submit()
}

function fillCentral() {
    rule_out_agi_by_field.val(2020)
    virtual_successes_field.val(1)
    regime_start_year_field.val(1956)
    first_trial_probability_field.val('1/300')
    g_exp_field.val(4.3)
    g_act_field.val(11)

    relative_imp_res_comp_field.val(5)
    comp_spending_assumption_field.val(1)


    init_weight_calendar_field.val(.3)
    init_weight_researcher_field.val(.3)
    init_weight_comp_relative_res_field.val(0.05)
    init_weight_lifetime_field.val(.10)
    init_weight_evolution_field.val(.15)
    init_weight_agi_impossible_field.val(.1)

    $('#main_form').submit()
}

function fillHigh() {
    rule_out_agi_by_field.val(2020)
    virtual_successes_field.val(1)
    regime_start_year_field.val(2000)
    first_trial_probability_field.val('1/100')
    g_exp_field.val(4.3)
    g_act_field.val(16)

    relative_imp_res_comp_field.val(5)
    comp_spending_assumption_field.val(100)


    init_weight_calendar_field.val(.1)
    init_weight_researcher_field.val(.4)
    init_weight_comp_relative_res_field.val(0.1)
    init_weight_lifetime_field.val(.1)
    init_weight_evolution_field.val(.2)
    init_weight_agi_impossible_field.val(.1)

    $('#main_form').submit()
}

function createTooltips(){
    // https://stackoverflow.com/a/58863161/

    $(".tooltip_anchor").each((index, element) => {
    let $anchor = $(element),
        $content = $anchor.attr('title');

    let timeoutId = null;
    function cancelClose() {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
    }
    function scheduleClose() {
      cancelClose();
      timeoutId = setTimeout(() => {
        $anchor.tooltip("close");
        timeoutId = null;
      }, 250)
    }

    let tooltipOptions = {
      content: () => $content,
      // Only the tooltip anchor should get a tooltip.  If we used "*", every child tag would
      //  also get a tooltip.
      items: ".tooltip_anchor",
      position: {
        my: "left center",
        at: "right+3 center",
        collision: "flipfit",
      },
    };
    if ($anchor.is(".tooltip_is_hoverable")) {
      $.extend(tooltipOptions, {
        open: (e) => {
          let $tooltip = $("[role='tooltip'],.ui-tooltip");
          $tooltip.on('mouseenter', cancelClose);
          $tooltip.on('mouseleave', scheduleClose);
        },
        tooltipClass: "hoverable_tooltip",
      });

      // Prevent jquery UI from closing the tooltip of an anchor with a hoverable tooltip.
      $anchor.off('mouseleave') // remove any previous event handlers
      $anchor.on('mouseleave', (e) => {
        // Instead, schedule a close.
        scheduleClose();
        e.stopImmediatePropagation();
      });
    }
    $anchor.tooltip(tooltipOptions);
  });
}