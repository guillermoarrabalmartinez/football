-- Table: public.player

-- DROP TABLE public.player;

CREATE TABLE public.player
(
    id integer NOT NULL GENERATED BY DEFAULT AS IDENTITY ( INCREMENT 1 START 1 MINVALUE 1 MAXVALUE 2147483647 CACHE 1 ),
    player_api_id integer,
    player_name text COLLATE pg_catalog."default",
    player_fifa_api_id integer,
    birthday text COLLATE pg_catalog."default",
    height integer,
    weight integer
)
WITH (
    OIDS = FALSE
)
TABLESPACE pg_default;

ALTER TABLE public.player
    OWNER to postgres;