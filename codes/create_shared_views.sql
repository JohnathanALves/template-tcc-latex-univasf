DELIMITER //
DROP PROCEDURE IF EXISTS create_shared_views;
CREATE PROCEDURE create_shared_views()
BEGIN 
	CALL create_view_alunos();
	CALL create_view_professores();
	CALL create_view_posts(); 
END //
DELIMITER ;
