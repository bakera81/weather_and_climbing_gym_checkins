/* set variables */
\set USERS _users
\set CHECKINS _checkins
\set USERS_TABLE :PREFIX:USERS
\set CHECKS_TABLE :PREFIX:CHECKINS

/* drop table if it exists and create new table */
BEGIN;

    DROP TABLE IF EXISTS public.:USERS_TABLE;

    CREATE TABLE :USERS_TABLE (
                    guid VARCHAR PRIMARY KEY,
                    home_gyms INT,
                    current_home_gym VARCHAR,
                    first_checkin TIMESTAMPTZ,
                    last_checkin TIMESTAMPTZ,
                    mem_checkins_count INT,
                    nmem_checkins_count INT,
                    tot_checkins_count INT,
                    min_age INT,
                    max_age INT
                    );

COMMIT;

/* insert aggregated values into users table */
BEGIN;

    INSERT INTO :USERS_TABLE (
                    guid,
                    home_gyms,
                    first_checkin,
                    last_checkin,
                    mem_checkins_count,
                    nmem_checkins_count,
                    tot_checkins_count,
                    min_age,
                    max_age)
                    SELECT
                        guid,
                        COUNT(DISTINCT home_gym),
                        MIN(datetime),
                        MAX(datetime),
                        COUNT(datetime) FILTER (WHERE checkin_type = 'Member'),
                        COUNT(datetime) FILTER (WHERE checkin_type = 'Non-Member'),
                        COUNT(datetime),
                        MIN(age_at_checkin),
                        MAX(age_at_checkin)
                    FROM 
                        :CHECKS_TABLE
                    GROUP BY
                        guid;
    
    /* create cte of home gyms at last checkin */
    WITH current_home_gyms AS (
                    SELECT
                        ct.guid,
                        ct.home_gym
                    FROM 
                        :USERS_TABLE ut
                    LEFT JOIN
                        :CHECKS_TABLE ct
                    ON
                        ct.datetime = ut.last_checkin
                        AND ct.guid = ut.guid)

    /* update current_home_gym column */
    UPDATE
        :USERS_TABLE ut
    SET 
        current_home_gym = chg.home_gym
    FROM
        current_home_gyms chg
    WHERE 
        ut.guid = chg.guid;

COMMIT;

/* add in cust table info*/
BEGIN;

    CREATE TEMPORARY TABLE customers (
                    id INT,
                    customer_id VARCHAR,
                    responsible_party_id VARCHAR,
                    customer_type VARCHAR,
                    address_1 VARCHAR,
                    address_2 VARCHAR,
                    city VARCHAR,
                    state VARCHAR,
                    zip VARCHAR,
                    belay VARCHAR,
                    first_contact_date VARCHAR,
                    pay_form VARCHAR,
                    status VARCHAR,
                    eft_dues FLOAT,
                    birthday TIMESTAMPTZ,
                    facility_access VARCHAR,
                    guid VARCHAR
                    );

    /*copy csv file into TABLE_NAME*/
    COPY customers(id,
                    customer_id,
                    responsible_party_id,
                    customer_type,
                    address_1,
                    address_2,
                    city,
                    state,
                    zip,
                    belay,
                    first_contact_date,
                    pay_form,
                    status,
                    eft_dues,
                    birthday,
                    facility_access,
                    guid)
    FROM :CSV_PATH DELIMITER ',' CSV HEADER;

    DELETE
        FROM
            customers a
        USING 
            customers b
        WHERE
            a.id < b.id
            AND a.guid = b.guid;

        /* create new table */
    CREATE TABLE new_:USERS_TABLE (
                    guid VARCHAR PRIMARY KEY,
                    home_gyms INT,
                    current_home_gym VARCHAR,
                    first_checkin TIMESTAMPTZ,
                    last_checkin TIMESTAMPTZ,
                    mem_checkins_count INT,
                    nmem_checkins_count INT,
                    tot_checkins_count INT,
                    min_age INT,
                    max_age INT,
                    customer_type VARCHAR,
                    address_1 VARCHAR,
                    address_2 VARCHAR,
                    city VARCHAR,
                    state VARCHAR,
                    zip VARCHAR,
                    belay VARCHAR,
                    status VARCHAR);

    /* insert ordered rows into new table */
    INSERT INTO new_:USERS_TABLE (
                        guid,
                        home_gyms,
                        first_checkin,
                        last_checkin,
                        mem_checkins_count,
                        nmem_checkins_count,
                        tot_checkins_count,
                        min_age,
                        max_age,
                        customer_type,
                        address_1,
                        address_2,
                        city,
                        state,
                        zip,
                        belay,
                        status
                        )
                    SELECT
                        c.guid,
                        u.home_gyms,
                        u.first_checkin,
                        u.last_checkin,
                        u.mem_checkins_count,
                        u.nmem_checkins_count,
                        u.tot_checkins_count,
                        u.min_age,
                        u.max_age,
                        c.customer_type,
                        c.address_1,
                        c.address_2,
                        c.city,
                        c.state,
                        c.zip,
                        c.belay,
                        c.status
                    FROM
                        :USERS_TABLE u
                    INNER JOIN
                        customers c
                    ON
                        u.guid = c.guid
                    ORDER BY
                        first_checkin;

    /* drop original table */
    DROP TABLE :USERS_TABLE;

    /* rename new table */
    ALTER TABLE new_:USERS_TABLE 
    RENAME TO :USERS_TABLE;


COMMIT;