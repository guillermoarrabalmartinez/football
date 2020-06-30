-- Table: public.shoton

-- DROP TABLE public.shoton;

CREATE TABLE public.shoton
(
    blocked text COLLATE pg_catalog."default",
    card_type text COLLATE pg_catalog."default",
    coordinates text COLLATE pg_catalog."default",
    del text COLLATE pg_catalog."default",
    elapsed integer,
    elapsed_plus integer,
    event_incident_typefk integer,
    goal_type text COLLATE pg_catalog."default",
    id integer,
    match_id integer,
    n integer,
    player1 integer,
    pos_x integer,
    pos_y integer,
    shoton text COLLATE pg_catalog."default",
    sortorder integer,
    stats text COLLATE pg_catalog."default",
    subtype text COLLATE pg_catalog."default",
    team integer,
    type text COLLATE pg_catalog."default",
    value text COLLATE pg_catalog."default"
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.shoton
    OWNER to postgres;