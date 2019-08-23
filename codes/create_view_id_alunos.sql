DELIMITER // 
DROP PROCEDURE IF EXISTS create_view_id_alunos;
CREATE PROCEDURE create_view_id_alunos
(
	IN base_table VARCHAR(30)
)
BEGIN

	SET @base = base_table;
	SET @view_base = CONCAT(@base, '_id_alunos');

	SET @query = CONCAT(
		' CREATE OR REPLACE VIEW ', 
		@view_base, 
		' AS  SELECT distinct(aluno_id) FROM ',
		@base,
		' ORDER BY aluno_id;'
	);

	PREPARE stmt FROM @query;
	EXECUTE stmt;

	DEALLOCATE PREPARE stmt;

END //
DELIMITER ;
