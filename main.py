def laplace(trials,failures):
	successes = trials - failures
	virtual_sucesses = 1
	next_trial_p = (virtual_sucesses+successes)/trials
	return next_trial_p

def noAIupdate(failures=2020-1956,forecast_years=16):
	p_failure_by_target = 1
	for i in range(forecast_years):
		p_failure = 1-laplace(failures,failures)
		p_failure_by_target = p_failure_by_target*p_failure
		failures += 1
	p_success_by_target = 1-p_failure_by_target
	return p_success_by_target