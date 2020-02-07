/* set variables */
\set COUNTS _counts
\set CHECKINS _checkins
\set COUNTS_TABLE :PREFIX:COUNTS
\set CHECKS_TABLE :PREFIX:CHECKINS

/* drop table if it exists and replace with new table */
BEGIN;

    DROP TABLE IF EXISTS public.:COUNTS_TABLE;

    CREATE TABLE :COUNTS_TABLE (
                    datetime TIMESTAMPTZ,
                    tot_checkins INT,
                    mem_checkins INT,
                    nmem_checkins INT
                    );


    INSERT INTO :COUNTS_TABLE
        SELECT
            datetime,
            COUNT(*) tot_checkins,
            COUNT(*) FILTER (WHERE checkin_type = 'Member') mem_checkins,
            COUNT(*) FILTER (WHERE checkin_type = 'Non-Member') nmem_checkins
        FROM
            :CHECKS_TABLE
        GROUP BY 
            datetime;

COMMIT;

BEGIN;

    /* Create new counts table to replace old one*/
    CREATE TABLE new_:COUNTS_TABLE (
                    datetime TIMESTAMPTZ,
                    tot_checkins INT,
                    mem_checkins INT,
                    nmem_checkins INT
                    );
    
    INSERT INTO new_:COUNTS_TABLE (
                    datetime,
                    tot_checkins,
                    mem_checkins,
                    nmem_checkins
                    )
            WITH checkin_days AS (
                    SELECT
                        DATE(datetime) date,
                        SUM(tot_checkins) tot_checkins_day
                    FROM
                        :COUNTS_TABLE
                    GROUP BY
                        DATE(datetime)
                    ORDER BY 
                        date
                    )
                    SELECT
                        a.datetime datetime,
                        (CASE 
                            WHEN tot_checkins IS NULL THEN 0
                            ELSE tot_checkins
                        END) tot_checkins,
                        (CASE 
                            WHEN mem_checkins IS NULL THEN 0
                            ELSE mem_checkins
                        END) mem_checkins,
                        (CASE 
                            WHEN nmem_checkins IS NULL THEN 0
                            ELSE nmem_checkins
                        END) nmem_checkins
                    FROM
                        (SELECT 
                            GENERATE_SERIES(MIN(datetime), MAX(datetime), '1h') datetime
                        FROM
                            :COUNTS_TABLE) a
                    LEFT JOIN
                        :COUNTS_TABLE c
                    ON 
                        c.datetime = a.datetime
                    LEFT JOIN
                        checkin_days cd
                    ON
                        cd.date = DATE(a.datetime)
                    WHERE
                        (
                            ( /* Tuesday, Wednesday, Thursday 6 am - 11 pm*/
                            (EXTRACT(hour from a.datetime) < 23
                                AND EXTRACT(hour from a.datetime) > 5)
                            AND EXTRACT(dow from a.datetime) in (2, 3, 4)
                            )
                            OR
                            ( /* Monday, Friday 6 am - 10 pm*/
                                (EXTRACT(hour from a.datetime) < 22
                                    AND EXTRACT(hour from a.datetime) > 5)
                                AND EXTRACT(dow from a.datetime) in (1, 5)
                            )
                            OR
                            ( /* Saturday 8 am - 8 pm*/
                                (EXTRACT(hour from a.datetime) < 20
                                    AND EXTRACT(hour from a.datetime) > 7)
                                AND EXTRACT(dow from a.datetime) = 6
                            )
                            OR
                            ( /* Sunday 8 am - 6 pm*/
                                (EXTRACT(hour from a.datetime) < 18
                                    AND EXTRACT(hour from a.datetime) > 7)
                                AND EXTRACT(dow from a.datetime) = 0
                            )
                        )
                        AND cd.tot_checkins_day > 50
                    ORDER BY
                        datetime;

    /* drop original table */
    DROP TABLE :COUNTS_TABLE;

    /* rename new table */
    ALTER TABLE new_:COUNTS_TABLE 
    RENAME TO :COUNTS_TABLE;

COMMIT;