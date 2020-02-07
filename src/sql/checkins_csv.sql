/* set variables */
    /* interval between checkins to determine duplicate checkins */
    \set dup_interval 4

/* drop table if it exists and replace with new table */
BEGIN;

    DROP TABLE IF EXISTS public.:TABLE_NAME;

    CREATE TABLE :TABLE_NAME (
                    date DATE,
                    checkin_hour INT,
                    facility_name VARCHAR,
                    customer_key VARCHAR,
                    guid VARCHAR,
                    age_at_checkin VARCHAR,
                    checkin_type VARCHAR,
                    checkin_status VARCHAR,
                    total_checkins VARCHAR
                    );

    /*copy csv file into TABLE_NAME*/
    COPY :TABLE_NAME(date, checkin_hour, facility_name, customer_key, guid, age_at_checkin, checkin_type, checkin_status, total_checkins)
    FROM :CSV_PATH DELIMITER ',' CSV HEADER;

COMMIT;

/* clean up table */
BEGIN;

    ALTER TABLE :TABLE_NAME
        ADD COLUMN datetime TIMESTAMPTZ,
        ADD COLUMN home_gym VARCHAR;

    UPDATE :TABLE_NAME
        SET datetime = date + interval '1h' * checkin_hour,
        home_gym = RIGHT(customer_key, 3);

    ALTER TABLE :TABLE_NAME
        DROP COLUMN date,
        DROP COLUMN checkin_hour,
        DROP COLUMN total_checkins;

    DELETE FROM :TABLE_NAME
        WHERE 
        guid = '-'
        OR age_at_checkin = '-';

COMMIT;

/* recreate table with checkin `id` and cast `age_at_checkin` to INT */
BEGIN;

    /* create new table */
    CREATE TABLE new_:TABLE_NAME (
                    id SERIAL PRIMARY KEY,
                    datetime TIMESTAMPTZ,
                    facility_name VARCHAR,
                    customer_key VARCHAR,
                    guid VARCHAR,
                    age_at_checkin INT,
                    checkin_type VARCHAR,
                    checkin_status VARCHAR,
                    home_gym VARCHAR);

    /* insert ordered rows into new table */
    INSERT INTO new_:TABLE_NAME (
                        datetime, 
                        facility_name, 
                        customer_key, 
                        guid, 
                        age_at_checkin, 
                        checkin_type, 
                        checkin_status,
                        home_gym)
                    SELECT
                        datetime, 
                        facility_name, 
                        customer_key, 
                        guid, 
                        age_at_checkin::INT, 
                        checkin_type, 
                        checkin_status,
                        home_gym
                    FROM
                        :TABLE_NAME
                    ORDER BY
                        datetime;

    /* drop original table */
    DROP TABLE :TABLE_NAME;

    /* rename new table */
    ALTER TABLE new_:TABLE_NAME 
    RENAME TO :TABLE_NAME;

COMMIT;

/* delete duplicates within 4 hours using an antijoin*/
BEGIN;

    CREATE TEMPORARY TABLE duplicates AS
        SELECT  
                t2.id dup_ids
            FROM 
                :TABLE_NAME AS t1 
            CROSS JOIN 
                :TABLE_NAME AS t2 
            WHERE 
                t1.datetime >= t2.datetime - interval '1h' * :dup_interval
                AND t1.guid = t2.guid 
                AND t1.id < t2.id;

    /* create new table */
    CREATE TABLE new_:TABLE_NAME (
                    id SERIAL PRIMARY KEY,
                    datetime TIMESTAMPTZ,
                    facility_name VARCHAR,
                    customer_key VARCHAR,
                    guid VARCHAR,
                    age_at_checkin INT,
                    checkin_type VARCHAR,
                    checkin_status VARCHAR,
                    home_gym VARCHAR
                        );

    /* insert non duplicates into table */
    INSERT INTO new_:TABLE_NAME (
                        id,
                        datetime, 
                        facility_name, 
                        customer_key, 
                        guid, 
                        age_at_checkin, 
                        checkin_type, 
                        checkin_status,
                        home_gym)
                    SELECT
                        t.id,
                        t.datetime, 
                        t.facility_name, 
                        t.customer_key, 
                        t.guid, 
                        t.age_at_checkin, 
                        t.checkin_type, 
                        t.checkin_status,
                        t.home_gym
                    FROM
                        :TABLE_NAME t
                    LEFT JOIN
                        duplicates d
                    ON
                        t.id = d.dup_ids
                    WHERE
                        d.dup_ids IS NULL
                    ORDER BY
                        t.id;

    /* drop original table */
    DROP TABLE :TABLE_NAME;

    /* rename new table */
    ALTER TABLE new_:TABLE_NAME 
    RENAME TO :TABLE_NAME;

COMMIT;


