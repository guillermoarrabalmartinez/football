-- Table: public.team_attributes

-- DROP TABLE public.team_attributes;

CREATE TABLE public.team_attributes
(
    id integer NOT NULL GENERATED BY DEFAULT AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 2147483647 CACHE 1 ),
    team_fifa_api_id integer,
    team_api_id integer,
    date text COLLATE pg_catalog."default",
    buildupplayspeed integer,
    buildupplayspeedclass text COLLATE pg_catalog."default",
    buildupplaydribbling integer,
    buildupplaydribblingclass text COLLATE pg_catalog."default",
    buildupplaypassing integer,
    buildupplaypassingclass text COLLATE pg_catalog."default",
    buildupplaypositioningclass text COLLATE pg_catalog."default",
    chancecreationpassing integer,
    chancecreationpassingclass text COLLATE pg_catalog."default",
    chancecreationcrossing integer,
    chancecreationcrossingclass text COLLATE pg_catalog."default",
    chancecreationshooting integer,
    chancecreationshootingclass text COLLATE pg_catalog."default",
    chancecreationpositioningclass text COLLATE pg_catalog."default",
    defencepressure integer,
    defencepressureclass text COLLATE pg_catalog."default",
    defenceaggression integer,
    defenceaggressionclass text COLLATE pg_catalog."default",
    defenceteamwidth integer,
    defenceteamwidthclass text COLLATE pg_catalog."default",
    defencedefenderlineclass text COLLATE pg_catalog."default"
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.team_attributes
    OWNER to postgres;