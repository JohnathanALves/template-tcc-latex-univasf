DELIMITER //
DROP FUNCTION IF EXISTS CALCULATE_PERIOD//
CREATE FUNCTION CALCULATE_PERIOD(first_semester VARCHAR(255), curr_semester VARCHAR(255)) RETURNS INTEGER DETERMINISTIC
BEGIN 
	SET @first_year = CAST(LEFT(first_semester,4)  AS UNSIGNED);
	SET @first_period = CAST(RIGHT(first_semester,1) AS UNSIGNED); 
	SET @curr_year =  CAST(LEFT(curr_semester,4)  AS UNSIGNED);
	SET @curr_period =  CAST(RIGHT(curr_semester,1) AS UNSIGNED); 
	return (@curr_year - @first_year) * 2 + @curr_period - @first_period + 1; 
END//
DELIMITER ;
