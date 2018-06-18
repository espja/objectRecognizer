#SET (CIMG_PROGRAMS )
SET (OPENCV_PROGRAMS )
SET (OPENCV_PROGRAMS2   find_orb DB_orb DB_surf DB_akaze find_objects)
#SET (MIXED_PROGRAMS )

SET(EXTERNAL_LIBS ${SYSTEM_LIBS})


foreach(program ${CIMG_PROGRAMS})
        add_executable(${program} ${program}.cpp)
		target_link_libraries(${program} ${EXTERNAL_LIBS})
endforeach(program)

foreach(program ${OPENCV_PROGRAMS})
        add_executable(${program} ${program}.cpp)
		target_link_libraries(${program} ${EXTERNAL_LIBS})
        target_link_libraries(${program} ${OpenCV_LIBS})       
endforeach(program)

foreach(program ${OPENCV_PROGRAMS2})
        add_executable(${program} ${program}.cpp hough.cpp)
        target_link_libraries(${program} ${OpenCV_LIBS})    
        target_link_libraries(${program} ${EXTERNAL_LIBS})   
endforeach(program)

foreach(program ${MIXED_PROGRAMS})
        add_executable(${program} ${program}.cpp)
		target_link_libraries(${program} ${EXTERNAL_LIBS})
        target_link_libraries(${program} ${OpenCV_LIBS})
endforeach(program)

install(TARGETS ${CIMG_PROGRAMS} ${OPENCV_PROGRAMS} ${OPENCV_PROGRAMS2} ${MIXED_PROGRAMS} DESTINATION ${CMAKE_INSTALL_BINDIR})




