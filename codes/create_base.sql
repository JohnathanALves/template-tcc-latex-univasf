DELIMITER //
DROP PROCEDURE IF EXISTS create_base;
CREATE PROCEDURE create_base
(
	IN base_table VARCHAR(30),
	IN first_semester VARCHAR(10),
	IN course_id BIGINT(10)
)
BEGIN
	SET @base = base_table;
	SET @first_semester = first_semester;
	SET @course_id = course_id;

	SET @query_drop_previous = CONCAT(
		'DROP TABLE IF EXISTS ', @base, ';'
	);

	PREPARE stmt FROM @query_drop_previous;
	EXECUTE stmt;
	DEALLOCATE PREPARE stmt;

	SET @query = CONCAT(
		' CREATE TABLE  ', @base,
		'  ( periodo int(11) , ',
		' disciplina_id bigint(10), ',
		'aluno_id bigint(10),',
		' PRIMARY KEY (aluno_id, disciplina_id, periodo) ',
		' ) ENGINE=InnoDB AS ',
		' SELECT ',
			' course.name AS \'curso\', ',
			' HANDLE_SEMESTER(semester.name) AS \'semestre\', ',
			' CALCULATE_PERIOD( ', @first_semester, ',HANDLE_SEMESTER(semester.name)) AS \'periodo\',',
			' discipline.fullname AS \'disciplina_nome\', ',
			' discipline.id  AS \'disciplina_id\', ',
			' CALCULATE_START_DATE(HANDLE_SEMESTER(semester.name), SUBSTRING_INDEX(\'', @base, '\', \'_\', 1)) AS \'data_inicio\', ',
			' CALCULATE_END_DATE(HANDLE_SEMESTER(semester.name), SUBSTRING_INDEX(\'', @base, '\', \'_\', 1)) AS \'data_fim\', ',
			' participant.id AS \'aluno_id\', ',
			' REMOVE_DOTS(DELETE_DOUBLE_SPACES(UCASE(CONCAT(participant.firstname, \' \',participant.lastname)))) AS \'aluno_nome\', ',
			' participant.username as \'cpf\' ',
		' FROM ',
			' mdl_course discipline ',
				' INNER JOIN ',
					' mdl_course_categories semester ',
				' ON ',
					' (discipline.category = semester.id) ',
				' INNER JOIN ',
					' mdl_course_categories course ',
				' ON ',
					' (course.id = semester.parent) ',
				' INNER JOIN ',
					' mdl_enrol enrol ',
				' ON ',
					' (enrol.courseid = discipline.id) ',
				' INNER JOIN ',
					' mdl_user_enrolments user_enrolments ',
				' ON ',
					' (user_enrolments.enrolid = enrol.id) ',
				' INNER JOIN ',
					' mdl_user participant ',
				' ON ',
					' (participant.id = user_enrolments.userid) ',
				' INNER JOIN ',
					' mdl_role_assignments rs ',
				' ON ',
					' (rs.userid = participant.id) ',
				' INNER JOIN ',
					' mdl_context e ',
				' ON ',
					' (e.id = rs.contextid AND discipline.id = e.instanceid) ',
		' WHERE ',
			' course.id =  ', @course_id,' AND',
			' e.contextlevel = 50 AND ',
			' rs.roleid = 5 AND ',
			' semester.name NOT REGEXP \'.*REOFERTA.*|.*REPERCURSO.*\' ',
		' ORDER BY ',
			' curso, ',
			' semestre, ',
			' periodo, ',
			' disciplina_id, ',
			' aluno_nome; '
	);

	PREPARE stmt FROM @query;
	EXECUTE stmt;
	DEALLOCATE PREPARE stmt;
END //
DELIMITER ;
