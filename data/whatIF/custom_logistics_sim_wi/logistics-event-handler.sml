(* Util *)
fun ints2strings([]) = [] |ints2strings(i::is) = Int.toString(i)::ints2strings(is)
fun nested_reverse([]) = [] | nested_reverse(l::ls) = (rev l)::nested_reverse(ls)
fun index_recursive([], _, _) = ~1 | index_recursive(y::xs, x, i) = if x=y then i else index_recursive(xs, x, i+1)
fun index([], _) = ~1 | index(xs, x) = index_recursive(xs, x, 0)
fun container_ids([]) = [] | container_ids((cr_id,_,_,_,_,_,_,_)::crs: Containers) = cr_id::container_ids(crs)
fun transport_document_ids_recursive([], ids) = ids | 
	transport_document_ids_recursive((_,td_id,_,_,_,_,_,_)::crs: Containers, ids) = 
		if mem ids td_id 
		then transport_document_ids_recursive(crs, ids)
		else transport_document_ids_recursive(crs, td_id::ids)
fun transport_document_ids(crs) = transport_document_ids_recursive(crs, [])

(* generic function for writing a list of strings to .csv *)
fun write_record(file_id, l) = 
let
   val file = TextIO.openAppend(file_id)
   val _ = TextIO.output(file, list2string(l))
   val _ = TextIO.output(file, "\n")
in
   TextIO.closeOut(file)
end;

(* write event to table "event" and respective event type table *)
fun write_event(event_id, et: EventType, ea_values: string list) = 
let
	val event_file_id = "./event.csv"
	val event_type_file_id = "./event_" ^ event_map_type(et) ^ ".csv"
	val date = t2s(now())
	val _ = write_record(event_file_id, [event_id, et])
	val _ = write_record(event_type_file_id, [event_id, date]^^ea_values)
in
   event_id
end;

(* write qualified relations to table "object_object" *)
fun write_relations_recursively(file, []) = TextIO.closeOut(file) | 
write_relations_recursively(file, [obj_id1, obj_id2, qualifier]::qualified_pairs) = 
   (TextIO.output(file, list2string([obj_id1, obj_id2, qualifier]));
   TextIO.output(file, "\n");
   write_relations_recursively(file, qualified_pairs));

fun cartesian_single(_,[],_) = [] | cartesian_single(obj_id1, obj_id2::obj_ids, qualifier) = [obj_id1, obj_id2, qualifier]::cartesian_single(obj_id1, obj_ids, qualifier)

fun relationship_cartesian([],_,_) = [] | relationship_cartesian(obj_id1::obj_ids, obj_ids2, qualifier) = cartesian_single(obj_id1, obj_ids2, qualifier)^^relationship_cartesian(obj_ids, obj_ids2, qualifier)

fun write_e2o_relations(qualified_pairs) =
let
   val file = TextIO.openAppend("./event_object.csv")
in
   write_relations_recursively(file, qualified_pairs)
end;

fun write_o2o_relations(qualified_pairs) =
let
   val file = TextIO.openAppend("./object_object.csv")
in
   write_relations_recursively(file, qualified_pairs)
end;

(* write object to table "object" and respective object type table *)
fun initialize_objects_recursively(object_file, object_type_file, object_type, []) = (TextIO.closeOut(object_file); TextIO.closeOut(object_type_file)) |
initialize_objects_recursively(object_file, object_type_file, object_type, (object_id::oa_values)::objects_with_attribute_values) =
let
	(*val time = tinit2s()*)
	val time = t2s(now())
	val changed_field = ""
in
   (TextIO.output(object_file, list2string([object_id, object_type]));
   TextIO.output(object_file, "\n");
   TextIO.output(object_type_file, list2string([object_id, time, changed_field]^^oa_values));
   TextIO.output(object_type_file, "\n");  
   initialize_objects_recursively(object_file, object_type_file, object_type, objects_with_attribute_values))
end;

fun initialize_objects(object_type, objects_with_attribute_values) = 
let
   val object_file_id = "./object.csv"
   val object_type_file_id = "./object_" ^ object_map_type(object_type) ^ ".csv"
   val object_file = TextIO.openAppend(object_file_id)
   val object_type_file = TextIO.openAppend(object_type_file_id)
in
   initialize_objects_recursively(object_file, object_type_file, object_type, objects_with_attribute_values)
end;


(***********************)
(* object initializers *)
(***********************)

fun initialize_vehicle((veh_id, idle_cap, td_ids, cr_ids, clock): Vehicle) = 
let
   val objects_with_attribute_values = [veh_id::[t2s(clock)]]
in
   if List.length td_ids > 0 orelse List.length cr_ids > 0 then initialize_objects("Vehicle", objects_with_attribute_values)
   else ()
end;

fun initialize_customer_order((co_id, nof_goods): CustomerOrder) = 
let
   val objects_with_attribute_values = [co_id::[Int.toString nof_goods]]
in
   initialize_objects("Customer Order", objects_with_attribute_values)
end;

fun initialize_transport_document((td_id, crs): TransportDocument) = 
let
   val objects_with_attribute_values = [td_id::[Int.toString (List.length crs)]]
in
   initialize_objects("Transport Document", objects_with_attribute_values)
end;

fun containers_attribute_values([]) = [] | containers_attribute_values((cr_id, _, _, nof_hus, _, cr_status, _, _)::crs: Containers) = 
let
	val status_str = if cr_status = empt then "empty" else "full" 
in 
	(cr_id::[Int.toString nof_hus, status_str])::containers_attribute_values(crs)
end;
fun initialize_containers(crs: Containers) = 
let
   val objects_with_attribute_values = containers_attribute_values(crs)
in
   initialize_objects("Container", objects_with_attribute_values)
end;

fun initialize_handling_unit((hu_id, cr_id): HandlingUnit) = 
let
   val objects_with_attribute_values = [hu_id::[]]
in
   initialize_objects("Handling Unit", objects_with_attribute_values)
end;

fun initialize_truck(tr_id: Truck) = 
let
   val objects_with_attribute_values = [tr_id::[]]
in
   initialize_objects("Truck", objects_with_attribute_values)
end;

fun initialize_forklift(fl_id: Forklift) = 
let
   val objects_with_attribute_values = [fl_id::[]]
in
   initialize_objects("Forklift", objects_with_attribute_values)
end;


(****************************)
(* object attribute updates *)
(****************************)

fun skipstrings(0) = [] | skipstrings(x) = ""::skipstrings(x-1)
fun update_object_attribute(object_type, object_id, changed_field, changed_value) =
let
   val object_type_file_id = "./object_" ^ object_map_type(object_type) ^ ".csv"
   val object_type_file = TextIO.openAppend(object_type_file_id)
   val time = t2s(now())
   val skips = skipstrings(index(oas_by_type(object_type), changed_field))
in
   (
   TextIO.output(object_type_file, list2string([object_id, time, changed_field]^^skips^^[changed_value]));
   TextIO.output(object_type_file, "\n");
   TextIO.closeOut(object_type_file)
   )
end;

fun update_cr_status((cr_id, _, _, _, _, cr_status, _, _): Container) = 
let 
	val status = if cr_status = full then "full" else "empty" 
in 
	update_object_attribute("Container", cr_id, "Status", status) 
end;
fun update_cr_weight(cr_id, weight) = update_object_attribute("Container", cr_id, "Weight", Real.toString weight)

(* Control status of transport documents concerning full departures (update if all containers have departed) *)		
fun decrement_pending_crs(td_id, (td_id2, nof_crs)::tdcs) = 
if td_id = td_id2 then 
	let 
		val new_nof_crs = nof_crs - 1
	in
		if new_nof_crs = 0 then 
			let 
				val _ = update_object_attribute("Transport Document", td_id, "Status", "shipped")	
			in 
				tdcs
			end
		else (td_id, nof_crs - 1)::tdcs
	end
else (td_id2, nof_crs)::decrement_pending_crs(td_id, tdcs);

fun update_departed_recursive([], tdcs) = tdcs | 
	update_departed_recursive((_, td_id, _, _, _, _, _, _)::crs, tdcs) = 
		update_departed_recursive(crs, decrement_pending_crs(td_id, tdcs))
fun update_departed((_, _, _, loaded_crs, _), tdcs) = update_departed_recursive(loaded_crs, tdcs)

fun updateTDStatus((td_id, _): TransportDocument, status) = update_object_attribute("Transport Document", td_id, "Status", status);

(**********)
(* Events *)
(**********)

fun write_register_order((co_id, nof_goods): CustomerOrder) =
let
	val event_id = "reg_"^co_id 
	val event_order = [[event_id, co_id, "registered CO"]]
	val e2o_relations = event_order
in
	(
	write_event(event_id, "Register Customer Order", []);
	initialize_customer_order((co_id, nof_goods));
	write_e2o_relations(e2o_relations)
	)
end;

fun write_create_document((co_id, nof_goods): CustomerOrder, td: TransportDocument) =
let
	val (td_id, crs) = td
	val event_id = "create_"^td_id 
	val event_order = [[event_id, co_id, "TD created for CO"]]
	val event_document = [[event_id, td_id, "created TD"]]
	val e2o_relations = event_order^^event_document
	val order_document = [[co_id, td_id, "TD for CO"]]
	val o2o_relations = order_document
in
	(
	write_event(event_id, "Create Transport Document", []);
	initialize_transport_document(td);
	write_e2o_relations(e2o_relations);
	write_o2o_relations(o2o_relations)
	)
end;

fun booked_vehicles_ids_recursive(highs, normals, []) = (highs, normals) | booked_vehicles_ids_recursive(highs, normals, (_, _, veh_id, _, _, _, _, cr_prio)::crs) = 
	if cr_prio = crp_high 
	then if mem highs veh_id then booked_vehicles_ids_recursive(highs, normals, crs) else booked_vehicles_ids_recursive(veh_id::highs, normals, crs)
	else if mem normals veh_id then booked_vehicles_ids_recursive(highs, normals, crs) else booked_vehicles_ids_recursive(highs, veh_id::normals, crs)
fun booked_vehicles_ids(crs) = booked_vehicles_ids_recursive([], [], crs)
fun write_book_vehicles(td: TransportDocument) = 
let
	val (td_id, crs) = td
	val (high_prio_veh_ids, normal_prio_veh_ids) = booked_vehicles_ids(crs)
	val veh_ids = high_prio_veh_ids^^normal_prio_veh_ids
	val event_id = "book_vehs_"^td_id 
	val event_td = [[event_id, td_id, "VHs booked for TD"]]
	val event_vehs = relationship_cartesian([event_id], veh_ids, "booked VHs")
	val td_vehs1 = relationship_cartesian([td_id], high_prio_veh_ids, "High-Prio VH for TD")
	val td_vehs2 = relationship_cartesian([td_id], normal_prio_veh_ids, "Regular VH for TD")
	val e2o_relations = event_td^^event_vehs
	val o2o_relations = td_vehs1^^td_vehs2
in
	(
	write_event(event_id, "Book Vehicles", []);
	write_e2o_relations(e2o_relations);
	write_o2o_relations(o2o_relations)
	)
end;

fun write_order_containers(td_id: TDId, crs: Containers) =
let
	val event_id = "order_crs_"^td_id 
	val cr_ids = container_ids(crs)
	val event_document = [[event_id, td_id, "ordered for TD"]]
	val event_containers = relationship_cartesian([event_id], cr_ids, "CRs ordered")
	val e2o_relations = event_document^^event_containers
	val document_containers = relationship_cartesian(cr_ids, [td_id], "CR for TD")
	val o2o_relations = document_containers
in
	(
	write_event(event_id, "Order Empty Containers", []);
	write_e2o_relations(e2o_relations);
	write_o2o_relations(o2o_relations)
	)
end;

fun write_pick_container((cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio): Container, tr_id: Truck) =
let
	val cr = (cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio)
	val event_id = "pick_"^cr_id 
	val event_container = [[event_id, cr_id, "CR picked"]]
	val event_truck = [[event_id, tr_id, "TR moved"]]
	val truck_container = [[tr_id, cr_id, "TR loads CR"]]
	val e2o_relations = event_container
	val o2o_relations = truck_container
in
	(
	initialize_containers([cr]);
	write_event(event_id, "Pick Up Empty Container", []);
	write_e2o_relations(e2o_relations);
	write_o2o_relations(o2o_relations)
	)
end;

fun write_collect_goods((hu_id, cr_id): HandlingUnit) =
let
	val hu = (hu_id, cr_id)
	val event_id = "collect_"^hu_id 
	val event_handling_unit = [[event_id, hu_id, "HU collected"]]
	val e2o_relations = event_handling_unit
in
	(
	initialize_handling_unit(hu);
	write_event(event_id, "Collect Goods", []);
	write_e2o_relations(e2o_relations)
	)
end;

(* Load Truck side effect *)
fun loadHU((cr_id, td_id, veh_id, nof_hus, hus, _, cr_weight, cr_prio): Container, (hu_id, cr_id2): HandlingUnit) = 
let
	val new_hus = (hu_id, cr_id)::hus
	val cr_status = if List.length new_hus = nof_hus then full else empt
in
	(cr_id, td_id, veh_id, nof_hus, new_hus, cr_status, cr_weight, cr_prio)
end;
fun write_load_truck((cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio): Container, tr_id: Truck, (hu_id, cr_id2): HandlingUnit) =
let
	val event_id = "load_truck_"^hu_id 
	val event_handling_unit = [[event_id, hu_id, "HU loaded"]]
	val event_container = [[event_id, cr_id, "CR laded"]]
	val event_truck = [[event_id, tr_id, "TR laded"]]
	val container_handling_unit = [[cr_id, hu_id, "CR contains HU"]]
	val e2o_relations = event_handling_unit^^event_container^^event_truck
	val o2o_relations = container_handling_unit
in
	(
	write_event(event_id, "Load Truck", []);
	write_e2o_relations(e2o_relations);
	write_o2o_relations(o2o_relations)
	)
end;

fun write_drive_to_terminal((cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio): Container, tr_id: Truck) =
let
	val event_id = "drive_term_"^cr_id 
	val event_container = [[event_id, cr_id, "CR moved"]]
	val event_truck = [[event_id, tr_id, "TR moved"]]
	val e2o_relations = event_container^^event_truck
in
	(
	write_event(event_id, "Drive to Terminal", []);
	write_e2o_relations(e2o_relations)
	)
end;

(* Weigh side effect *)
fun sample_weight_recursive(0) = 0.0 | sample_weight_recursive(nof_hus) = sample_weight_recursive(nof_hus - 1) + 
let
	val b = uniform(0.0,1.0)
in
	if b < 0.5 then 200.0
	else if b < 0.9 then 180.0
	else 250.0
end;
fun sample_weight(nof_hus) = sample_weight_recursive(nof_hus)
fun write_weigh((cr_id, td_id, veh_id, nof_hus, hus, cr_status, _, cr_prio): Container, fl_id: Forklift) =
let
	val event_id = "weigh_"^cr_id
	val weight = sample_weight(nof_hus)
	val event_container = [[event_id, cr_id, "CR weighted"]]
	val event_forklift = [[event_id, fl_id, "FL weighing"]]
	val e2o_relations = event_container^^event_forklift
	val _ = (
		write_event(event_id, "Weigh", []);
		write_e2o_relations(e2o_relations);
		update_cr_weight(cr_id, weight)
	)
in
	(* Return updated container object as side effect *)
	(cr_id, td_id, veh_id, nof_hus, hus, cr_status, weight, cr_prio)
end;

fun write_place_in_stock((cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio): Container, fl_id: Forklift) =
let
	val event_id = "stock_"^cr_id 
	val event_container = [[event_id, cr_id, "CR stored"]]
	val event_forklift = [[event_id, fl_id, "FL moved"]]
	val e2o_relations = event_container^^event_forklift
in
	(
	write_event(event_id, "Place in Stock", []);
	write_e2o_relations(e2o_relations)
	)
end;

fun write_bring_to_bay((cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio): Container, fl_id: Forklift) =
let
	val event_id = "to_bay_"^cr_id 
	val event_container = [[event_id, cr_id, "CR brought to bay"]]
	val event_forklift = [[event_id, fl_id, "FL moved"]]
	val e2o_relations = event_container^^event_forklift
in
	(
	write_event(event_id, "Bring to Loading Bay", []);
	write_e2o_relations(e2o_relations)
	)
end;

fun getVehicleById(_, []) = ERROR_VEHICLE() | getVehicleById(veh_id, (oid, idle_cap, scheduled_cr_ids, loaded_crs, clock)::vehs) = 
if veh_id = oid then (veh_id, idle_cap, scheduled_cr_ids, loaded_crs, clock) else getVehicleById(veh_id, vehs);

fun updateVH(new_vh, checked, []) = checked | updateVH(new_vh: Vehicle, checked: Vehicles, old_vh::unchecked: Vehicles) =
if (#1 new_vh = #1 old_vh) then checked^^[new_vh]^^unchecked 
else updateVH(new_vh, checked^^[old_vh], unchecked)
fun write_load_to_vehicle((cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio): Container, fl_id: Forklift, vehs: Vehicles) =
let
	val cr = (cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio)
	val (oid, idle_cap, scheduled_cr_ids, loaded_crs, clock) = getVehicleById(veh_id, vehs)
	val missed = (oid = "errorVH")
	val vehs2 = 
		if missed then (* missed: reschedule container *) vehs
		else let
			(* proper loading event *)
			val event_id = "load_veh_"^cr_id 
			val event_container = [[event_id, cr_id, "CR loaded"]]
			val event_forklift = [[event_id, fl_id, "FL moved"]]
			val event_vehicle = [[event_id, veh_id, "VH laded"]]
			val e2o_relations = event_container^^event_forklift^^event_vehicle
			val _ = (
				write_event(event_id, "Load to Vehicle", []);
				write_e2o_relations(e2o_relations)
			)
			val new_vh = (veh_id, idle_cap, scheduled_cr_ids, cr::loaded_crs, clock)
			val vehs2 = updateVH(new_vh, [], vehs)
		in
			vehs2
		end
in
	(missed, vehs2)
end;

fun write_reschedule_container(cr: Container, vehs: Vehicles) =
let
	val (high_prio_crs, _, vehs) = assign_vehicle_prio_high([cr], vehs, [])
	val reassigned_cr = List.nth(high_prio_crs, 0)
	val (cr_id, td_id, veh_id, nof_hus, hus, cr_status, cr_weight, cr_prio) = reassigned_cr
	val event_id = "resch_cr_"^cr_id
	val event_container = [[event_id, cr_id, "CR rescheduled"]]
	val event_vehicle = [[event_id, veh_id, "booked VH"]]
	val event_td = [[event_id, td_id, "TD with CR rescheduled"]]
	val td_vehicle = [[td_id, veh_id , "Ersatz VH for TD"]]
	val e2o_relations = event_container^^event_vehicle^^event_td
	val o2o_relations = td_vehicle
	val _ = (
		write_event(event_id, "Reschedule Container", []);
		write_e2o_relations(e2o_relations);
		write_o2o_relations(o2o_relations)
	)
in
	(reassigned_cr, vehs)
end;

fun write_depart((veh_id, idle_cap, scheduled_cr_ids, loaded_crs, clock): Vehicle) =
let
	val event_id = "depart_"^veh_id 
	val event_vehicle = [[event_id, veh_id, "VH departed"]]
	val cr_ids = container_ids(loaded_crs)
	val td_ids = transport_document_ids(loaded_crs)
	val event_containers = relationship_cartesian([event_id], cr_ids, "CR departed")
	val event_tds = relationship_cartesian([event_id], td_ids, "TD with CR departure")
	val e2o_relations = event_vehicle^^event_containers^^event_tds
in
	if List.length cr_ids > 0 then (
	write_event(event_id, "Depart", []);
	write_e2o_relations(e2o_relations)
	)
	else ()
end;
