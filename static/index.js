$(document).ready(function() {

// Create jQuery objects once, use multiple times later
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

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Hack to move y-axes that were being cut off. Customizing axis placement appears to be impossible or poorly documented in mpld3
async function moveYAxes(attempts){
    labels = $('.mpld3-text')
    console.log(attempts,labels.length)
    if (labels.length > 0) {
        labels.attr('transform', (i, value) => `${value || ""} translate(0 10)`)
        return // after one attempt on loaded axes
    }
    else if (attempts < 20) {
        await sleep(200)
        attempts++
        moveYAxes(attempts)
    }
}
moveYAxes(0)

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

    $('form').submit()
}

function fillCentral() {
    rule_out_agi_by_field.val(2020)
    virtual_successes_field.val(1)
    regime_start_year_field.val(1956)
    first_trial_probability_field.val('1/300')
    g_exp_field.val(4.3)
    g_act_field.val(11)

    relative_imp_res_comp_field.val(5)
    comp_spending_assumption_field.val(1000)


    init_weight_calendar_field.val(.3)
    init_weight_researcher_field.val(.3)
    init_weight_comp_relative_res_field.val(0.05)
    init_weight_lifetime_field.val(.10)
    init_weight_evolution_field.val(.15)
    init_weight_agi_impossible_field.val(.1)

    $('form').submit()
}

function fillHigh() {
    rule_out_agi_by_field.val(2020)
    virtual_successes_field.val(1)
    regime_start_year_field.val(2000)
    first_trial_probability_field.val('1/100')
    g_exp_field.val(4.3)
    g_act_field.val(16)

    relative_imp_res_comp_field.val(5)
    comp_spending_assumption_field.val(100000)


    init_weight_calendar_field.val(.1)
    init_weight_researcher_field.val(.4)
    init_weight_comp_relative_res_field.val(0.1)
    init_weight_lifetime_field.val(.1)
    init_weight_evolution_field.val(.2)
    init_weight_agi_impossible_field.val(.1)

    $('form').submit()
}