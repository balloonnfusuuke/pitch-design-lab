/** 5.125 oz baseball. */
export const BALL_MASS_KG = 0.145;

/** Midpoint of the 9.00–9.25 inch rule. */
export const BALL_CIRCUMFERENCE_M = 9.125 * 0.0254;

export const BALL_RADIUS_M = BALL_CIRCUMFERENCE_M / (2 * Math.PI);

export const BALL_AREA_M2 = Math.PI * BALL_RADIUS_M * BALL_RADIUS_M;

/**
 * I ≈ 2/5 m r². A baseball is not a uniform solid sphere;
 * this is the usual first approximation, and the release model
 * absorbs the difference in calibrated finger force.
 */
export const BALL_INERTIA_KG_M2 = 0.4 * BALL_MASS_KG * BALL_RADIUS_M * BALL_RADIUS_M;

export const GRAVITY_MS2 = 9.80665;

export const RUBBER_TO_PLATE_M = 60.5 * 0.3048;

export const PLATE_HALF_WIDTH_M = (17 * 0.0254) / 2;

export const MPH_TO_MS = 0.44704;

export const MS_TO_MPH = 1 / MPH_TO_MS;

export const M_TO_IN = 39.37007874015748;

export const M_TO_FT = 3.280839895013123;

export const FT_TO_M = 0.3048;

/** About 1% of spin is lost over a typical 0.4 s flight. */
export const SPIN_DECAY_TAU_S = 30;

/**
 * Nathan, May 2020 update of the Statcast spin-axis notes:
 * CL = A (1 − exp(−B S)), S = r ω_transverse / v.
 * A = 0.336, B = 6.041.
 */
export const NATHAN_CL_A = 0.336;
export const NATHAN_CL_B = 6.041;
