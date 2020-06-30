-- Table: public.shotoff

-- DROP TABLE public.shotoff;

CREATE TABLE public.shotoff
(
    card_type text COLLATE pg_catalog."default",
    coordinates text COLLATE pg_catalog."default",
    del text COLLATE pg_catalog."default",
    elapsed integer,
    elapsed_plus integer,
    event_incident_typefk integer,
    id integer,
    match_id integer,
    n integer,
    player1 integer,
    pos_x integer,
    pos_y integer,
    shotoff text COLLATE pg_catalog."default",
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

ALTER TABLE public.shotoff
    OWNER to postgres;