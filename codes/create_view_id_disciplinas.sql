DELIMITER // 
DROP PROCEDURE IF EXISTS create_view_id_disciplinas;
CREATE PROCEDURE create_view_id_disciplinas
(
	IN base_table VARCHAR(30)
)
BEGIN

	SET @base = base_table;
	SET @view_base = CONCAT(@base, '_id_disciplinas');

	SET @query = CONCAT(
		'CREATE OR REPLACE VIEW ',
		@view_base,
		' AS SELECT distinct(disciplina_id) FROM ', 
		@base,
		' ORDER BY disciplina_id; '	
	);

	PREPARE stmt FROM @query;
	EXECUTE stmt;

	DEALLOCATE PREPARE stmt;

END //
DELIMITER ;
