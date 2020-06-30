-- Table: public.possesion

-- DROP TABLE public.possesion;

CREATE TABLE public.possesion
(
    awaypos integer,
    card_type text COLLATE pg_catalog."default",
    comment integer,
    del integer,
    elapsed integer,
    elapsed_plus integer,
    event_incident_typefk integer,
    goal_type text COLLATE pg_catalog."default",
    homepos integer,
    id integer,
    injury_time integer,
    match_id integer,
    n integer,
    pos_x text COLLATE pg_catalog."default",
    pos_y text COLLATE pg_catalog."default",
    possession text COLLATE pg_catalog."default",
    sortorder integer,
    stats text COLLATE pg_catalog."default",
    subtype text COLLATE pg_catalog."default",
    type text COLLATE pg_catalog."default",
    value text COLLATE pg_catalog."default"
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.possesion
    OWNER to postgres;