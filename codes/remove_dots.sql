DELIMITER //
DROP FUNCTION IF EXISTS REMOVE_DOTS//
CREATE FUNCTION REMOVE_DOTS(str VARCHAR(255)) RETURNS VARCHAR(255) DETERMINISTIC
BEGIN
	while instr(str, '.') > 0 do
		set str = replace(str, '.', '');
	end while;
	return str;
END//
DELIMITER ;
