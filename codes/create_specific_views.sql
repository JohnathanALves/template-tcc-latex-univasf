DELIMITER //
DROP PROCEDURE IF EXISTS create_specific_views;
CREATE PROCEDURE create_specific_views(
	IN base VARCHAR(255)
)
BEGIN
	CALL create_view_disciplinas(base);
	CALL create_view_id_alunos(base);
	CALL create_view_id_disciplinas(base);
	CALL create_views_log_reduzido(base);
END //
DELIMITER ;
