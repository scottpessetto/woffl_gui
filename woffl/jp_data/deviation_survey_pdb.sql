-- One survey per well: the PREFERRED one. The view is a HISTORY view - it
-- carries every survey version (MPC-05 had 12 survey_nums, one of them with
-- TVD = 0), and pulling without this filter stacked them into one CSV with
-- duplicated depths and impossible TVD steps (review 2026-09-01, FLOW-3;
-- verified 2026-09-02: all 887 MILNEPT wells have exactly one preferred
-- survey, none with duplicate depths). SURVEY_NUM is kept for provenance.
SELECT API, SW_NAME, BU_ID, PROJECT, SURVEY_NUM, AZIMUTH, DOG_LEG, INCLINATION, MEAS_DEPTH, SUBSEA_DEPTH, TVD_DEPTH
FROM PUBLIC_PROJECT.DEV_SURVEY_HIST_DEPTH_TVD_VEW
WHERE BU_ID = 'MILNEPT'
AND PREFERRED_FLAG = 'Y'
AND SW_NAME LIKE :param
