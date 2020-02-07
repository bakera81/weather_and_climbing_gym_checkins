/* drop table if it exists and replace with new table */
BEGIN;

DROP TABLE IF EXISTS public.:TABLE_NAME;

CREATE TABLE :TABLE_NAME (
                date DATE,
                hour INT,
                dry_bulb_temp_c FLOAT,
                total_cloud_cover FLOAT,
                opaque_cloud_cover FLOAT,
                wind_speed_6ft_m_s FLOAT,
                wind_speed_19ft_m_s FLOAT,
                precipitation_mm FLOAT,
                snow_depth_cm FLOAT,
                snow_meas_quality FLOAT
                );

/*copy csv file into weather table*/
COPY :TABLE_NAME
FROM :CSV_PATH DELIMITER ',' CSV HEADER;

COMMIT;



/* clean up table */
BEGIN;

    /* Add columns for clean data */
    ALTER TABLE :TABLE_NAME
        ADD COLUMN datetime TIMESTAMPTZ,
        ADD COLUMN temp_f FLOAT,
        ADD COLUMN cloud_cover FLOAT,
        ADD COLUMN wind_lo_mph FLOAT,
        ADD COLUMN wind_hi_mph FLOAT,
        ADD COLUMN precip_in FLOAT,
        ADD COLUMN snow_depth_in FLOAT;

    /* convert date and hour columns to datetime and convert units to empircal units */
    UPDATE :TABLE_NAME
        SET 
            datetime = date + interval '1h' * hour,
            temp_f = (dry_bulb_temp_c * (9.0 / 5.0)) + 32.0,
            cloud_cover = (CASE
                WHEN opaque_cloud_cover < 0 THEN NULL
                ELSE opaque_cloud_cover
                        END),
            wind_lo_mph = wind_speed_6ft_m_s * 2.237,
            wind_hi_mph = wind_speed_19ft_m_s * 2.237,
            precip_in = (precipitation_mm / 25.4) * 60;

COMMIT;

/* clean and impute snow depth data */
BEGIN;
    /* create function to convert timestamp to float type */
    CREATE OR REPLACE FUNCTION public.timestamp_to_seconds(timestamp_t TIMESTAMPTZ)
        RETURNS FLOAT AS $$
            SELECT EXTRACT(epoch from timestamp_t)
        $$ LANGUAGE SQL;

    /* create linear interpolation function */
    CREATE OR REPLACE FUNCTION public.linear_interpolate(x_i TIMESTAMPTZ, x_0 TIMESTAMPTZ, y_0 FLOAT, x_1 TIMESTAMPTZ, y_1 FLOAT)
    RETURNS FLOAT AS $$
        SELECT (($5 - $3) / (public.timestamp_to_seconds($4) - public.timestamp_to_seconds($2))) 
        * (public.timestamp_to_seconds($1) - public.timestamp_to_seconds($2)) 
        + $3;
    $$ LANGUAGE SQL;

/* create temporary tables for cleaning snow_depth */
    /* create temp table based on null rules */
    CREATE TEMPORARY TABLE snow_depth_nulled AS (
        SELECT
                datetime,
                (CASE 
                    WHEN snow_depth_cm > 1 AND snow_depth_cm < 40 THEN snow_depth_cm / 2.54 /* valid values */
                    WHEN snow_depth_cm < 1 AND snow_depth_cm > -1 THEN 0 /* sensor specs +/- 1 cm accuracy */
                    WHEN ABS(snow_depth_cm - LAG(snow_depth_cm) OVER (ORDER BY datetime)) > 10 THEN NULL
                    ELSE NULL
                END) AS snow_depth_in
        FROM :TABLE_NAME
    );
    
    /* create temp table for interpolation endoints */
    CREATE TEMPORARY TABLE interp AS (
        SELECT
            datetime, 
            snow_depth_in, 
            snow_depth_partition,
            FIRST_VALUE(snow_depth_in) OVER (PARTITION BY snow_depth_partition ORDER BY datetime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) first_sd,
            FIRST_VALUE(datetime) OVER (PARTITION BY snow_depth_partition ORDER BY datetime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) first_ts,
            LAST_VALUE(datetime) OVER (PARTITION BY snow_depth_partition ORDER BY datetime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) + interval '1h' last_ts
        FROM (
            SELECT
                datetime,
                snow_depth_in,
                SUM(CASE WHEN snow_depth_in IS NULL THEN 0 ELSE 1 END) OVER (ORDER BY datetime) snow_depth_partition
            FROM snow_depth_nulled
            ORDER BY datetime ASC
            ) q1
    );

    /* create add last snow depth to interpolate to temp table */
    CREATE TEMPORARY TABLE interpolation_station AS (
        SELECT
            i.datetime datetime,
            i.snow_depth_in snow_depth_in,
            snow_depth_partition,
            first_sd,
            s.snow_depth_in last_sd,
            first_ts,
            last_ts
        FROM
            interp i
        LEFT JOIN
            snow_depth_nulled s
        ON
            i.last_ts = s.datetime
        ORDER BY datetime
    );
        
    UPDATE :TABLE_NAME t
        SET 
            snow_depth_in = (
            CASE
                WHEN t.snow_depth_in IS NULL
                    THEN public.linear_interpolate(i.datetime, i.first_ts, i.first_sd, i.last_ts, i.last_sd)
                ELSE t.snow_depth_in
            END)
        FROM interpolation_station i
        WHERE i.datetime = t.datetime;

COMMIT;

/* BEGIN;


    /* drop unused columns */
    -- ALTER TABLE :TABLE_NAME
    --     DROP COLUMN total_cloud_cover,
    --     DROP COLUMN opaque_cloud_cover,
    --     DROP COLUMN date,
    --     DROP COLUMN hour,
    --     DROP COLUMN dry_bulb_temp_c,
    --     DROP COLUMN wind_speed_6ft_m_s,
    --     DROP COLUMN wind_speed_19ft_m_s,
    --     DROP COLUMN wind_hi_mph,
    --     DROP COLUMN precipitation_mm,
    --     DROP COLUMN snow_depth_cm;

COMMIT; */

