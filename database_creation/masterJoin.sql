--match shots
select*from(
--MATCH JOINED WITH CORNERS
select*from(
--TABLE A = MATCHES
        select*from match as a 
left join
--TABLE B = CORNERS
            (select match_id, count(id) as corners from corner_detail group by match_id) as b
on a.id = b.match_id) as x
left join
--TABLE c = POSSESION
                    (select match_id,awaypos,homepos, max(elapsed) minuto from possesion group by match_id) as c
on x.id = c.match_id) as z
left join
--TABLE Y = SHOTSONYOFF
                            (select v.match_id, shotson, shotsoff from (select match_id, count(id) as shotson from shoton group by match_id) as v
                            left join (select match_id, count(id) as shotsoff from shotoff group by match_id) as w
                            on v.match_id = w.match_id) as y
on z.id = y.match_id
                            
